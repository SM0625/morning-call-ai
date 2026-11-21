package com.morningcall.app

import android.content.Context
import com.google.ai.edge.litertlm.Backend
import com.google.ai.edge.litertlm.Content
import com.google.ai.edge.litertlm.Contents
import com.google.ai.edge.litertlm.Conversation
import com.google.ai.edge.litertlm.ConversationConfig
import com.google.ai.edge.litertlm.Engine
import com.google.ai.edge.litertlm.EngineConfig
import com.google.ai.edge.litertlm.Message
import com.google.ai.edge.litertlm.MessageCallback
import com.google.ai.edge.litertlm.SamplerConfig
import com.google.ai.edge.litertlm.ThinkingConfig
import org.json.JSONArray
import org.json.JSONObject
import java.io.File
import java.security.MessageDigest
import java.io.FileOutputStream
import java.net.HttpURLConnection
import java.net.URL
import java.util.concurrent.Executors
import java.util.concurrent.LinkedBlockingQueue
import java.util.concurrent.TimeUnit

/**
 * 폰 안에서 Gemma 4 E2B를 돌린다 (Google LiteRT-LM). PC 서버·와이파이가 꺼져 있어도 깨울 수 있게 하려는 것.
 * 모델(약 2.6GB)은 처음 한 번 Hugging Face에서 받아 앱 저장소에 둔다. 받는 모델은 깨우기 대화로 미세조정한 것이다.
 * 대화 엔진(engine/dialog.py)이 [start]로 토큰 스트림을 받아 쓴다 (engine/phone.py).
 *
 * 서비스용:
 * - 어떤 모델을 받을지는 저장소의 manifest.json이 정한다 {"version", "file", "size", "sha256"}.
 *   새 모델을 올리고 version만 올리면, 앱을 다시 배포하지 않아도 "새 모델 받기"가 뜬다.
 * - 다 받으면 크기와 SHA-256을 확인한 뒤에만 쓴다 (깨진 파일로 알람이 실패하지 않게).
 * - 새 모델로 바꾸면 예전 모델 파일은 지워 저장 공간을 돌려준다.
 */
object PhoneAi {
    const val MODEL_NAME = "gemma-4-E2B 미세조정 (폰)"
    // 미세조정 모델 저장소 (tools/finetune/kaggle_convert_ple.ipynb로 만든 .litertlm + manifest.json)
    private const val REPO = "tkdals24/morning-gemma-litertlm"
    private const val MANIFEST_URL = "https://huggingface.co/$REPO/resolve/main/manifest.json"
    private fun fileUrl(file: String) = "https://huggingface.co/$REPO/resolve/main/$file"
    private const val PREFS = "phone_ai"
    // 서버(main.py GENERATION_OPTIONS)와 같은 생성 설정
    private const val TEMPERATURE = 0.7
    private const val MAX_OUTPUT_TOKENS = 120
    private const val TOKEN_TIMEOUT_S = 60L
    private const val LOAD_TIMEOUT_MS = 90_000L
    // 앞 대답의 남은 생성(최대 120토큰, 약 2초)을 기다리는 시간. 보통은 AI가 말하는 동안 이미 끝나 있다
    private const val PREVIOUS_WAIT_MS = 3_000L

    /** none(모델 없음) | downloading | verifying(받은 파일 확인 중) | downloaded(받았지만 아직 안 올림) | loading | ready | error */
    @Volatile var status = "none"
        private set
    /** 받는 중이면 진행률(%), 오류면 내용, 준비되면 쓰는 장치(GPU/CPU) */
    @Volatile var detail = ""
        private set
    /** 상태가 바뀔 때마다 화면에 알린다 (MainActivity) */
    var onStatus: ((String, String) -> Unit)? = null

    private var engine: Engine? = null
    private var current: TokenStream? = null
    private val loader = Executors.newSingleThreadExecutor()

    private fun setStatus(s: String, d: String = "") {
        status = s
        detail = d
        onStatus?.invoke(s, d)
    }

    private fun modelsDir(ctx: Context) = File(ctx.filesDir, "models")
    private fun prefs(ctx: Context) = ctx.getSharedPreferences(PREFS, Context.MODE_PRIVATE)

    /** 받아서 확인까지 끝난 모델 파일 (없으면 null). 예전 버전 앱이 받은 공식 모델도 그대로 쓴다 */
    private fun installedModel(ctx: Context): File? {
        prefs(ctx).getString("file", null)?.let { File(modelsDir(ctx), it) }?.takeIf { it.exists() }?.let { return it }
        return File(modelsDir(ctx), "gemma-4-E2B-it.litertlm").takeIf { it.exists() }
    }
    private fun installedVersion(ctx: Context) = prefs(ctx).getInt("version", 0)

    /**
     * 개발용: PC에서 adb로 앱 폴더에 넣으면 받은 모델 대신 쓴다 (새로 변환한 모델을 올리기 전에 시험할 때)
     * adb push morning-gemma.litertlm /sdcard/Android/data/com.morningcall.app/files/
     */
    private fun devModel(ctx: Context) =
        ctx.getExternalFilesDir(null)?.let { File(it, "morning-gemma.litertlm") }?.takeIf { it.exists() }

    /** 통화 기록에 남기는 모델 이름 (어느 모델·버전으로 통화했는지 구분) */
    var modelName = MODEL_NAME
        private set

    // ─── 저장소의 최신 모델 정보 (manifest.json) ─────────────────────────────────
    data class Manifest(val version: Int, val file: String, val size: Long, val sha256: String)
    @Volatile private var latest: Manifest? = null

    /** 받을 크기(MB). 화면의 "받기" 버튼 옆에 보여 준다 (모르면 0) */
    val downloadSizeMb get() = (latest?.size ?: 0L) / 1_000_000
    /** 받아 둔 모델보다 새 버전이 저장소에 있으면 true */
    @Volatile var updateAvailable = false
        private set
    /** 새 모델 받기에 실패했을 때 이유 (지금 모델로 계속 쓴다) */
    @Volatile var updateError = ""
        private set
    /** 새 모델을 받는 중이면 0~100(%), 확인 중이면 101, 아니면 -1 */
    @Volatile var updateProgress = -1
        private set

    private fun fetchManifest(): Manifest {
        val conn = (URL(MANIFEST_URL).openConnection() as HttpURLConnection).apply {
            connectTimeout = 10_000
            readTimeout = 10_000
        }
        if (conn.responseCode != HttpURLConnection.HTTP_OK) error("모델 정보를 받지 못했어요 (서버 응답 ${conn.responseCode})")
        val j = JSONObject(conn.inputStream.bufferedReader().readText())
        return Manifest(j.getInt("version"), j.getString("file"), j.getLong("size"), j.getString("sha256").lowercase())
    }

    /** 앱을 열 때 뒤에서 확인한다. 인터넷이 없으면 조용히 넘어간다 (받아 둔 모델로 계속 동작) */
    private fun checkUpdate(ctx: Context) = Thread {
        runCatching { fetchManifest() }.onSuccess { m ->
            latest = m
            updateAvailable = installedModel(ctx) != null && m.version > installedVersion(ctx)
            setStatus(status, detail)  // 화면에 "새 모델" 표시를 다시 그리게 한다
        }
    }.start()

    /** 앱을 열 때: 받아 둔 모델이 있으면 바로 메모리에 올린다 (알람이 울릴 때 기다리지 않도록) */
    fun init(ctx: Context) {
        if (status == "none" && (devModel(ctx) != null || installedModel(ctx) != null)) setStatus("downloaded")
        load(ctx)
        checkUpdate(ctx)
    }

    // ─── 모델 받기 ───────────────────────────────────────────────────────────────
    /**
     * 처음 받기와 새 모델 받기 모두 이것을 쓴다. 이어 받기를 지원한다(끊겨도 다시 누르면 받은 곳부터).
     * 새 모델을 받는 동안에도 지금 모델로 계속 통화할 수 있게, 다 받고 확인한 뒤에만 바꾼다.
     */
    @Volatile private var downloading = false
    fun download(ctx: Context) {
        if (downloading || status == "loading") return
        if (installedModel(ctx) != null && !updateAvailable && status != "error") return
        downloading = true
        updateError = ""
        // 이미 모델이 있으면(새 모델 받기) 상태는 그대로 두고 진행률만 따로 알린다. 받는 동안에도 지금 모델로 통화한다
        val wasReady = status == "ready"
        fun progress(p: Int) {
            if (wasReady) { updateProgress = p; setStatus(status, detail) }
            else if (p > 100) setStatus("verifying") else setStatus("downloading", "$p")
        }
        progress(0)
        Thread {
            try {
                val m = latest ?: fetchManifest().also { latest = it }
                val dir = modelsDir(ctx).apply { mkdirs() }
                val target = File(dir, m.file)
                val part = File(dir, "${m.file}.v${m.version}.part")
                // 다른 버전을 받다 만 조각은 지운다
                dir.listFiles { f -> f.name.endsWith(".part") && f != part }?.forEach { it.delete() }
                val done = if (part.exists()) part.length() else 0L
                if (done < m.size) {
                    if (ctx.filesDir.usableSpace < m.size - done + 200_000_000L)
                        error("저장 공간이 부족해요 (${m.size / 1_000_000_000.0}GB 필요)".replace(Regex("(\\.\\d)\\d+"), "$1"))
                    val conn = (URL(fileUrl(m.file)).openConnection() as HttpURLConnection).apply {
                        connectTimeout = 15_000
                        readTimeout = 30_000
                        if (done > 0) setRequestProperty("Range", "bytes=$done-")
                    }
                    val resumed = conn.responseCode == HttpURLConnection.HTTP_PARTIAL
                    if (!resumed && conn.responseCode != HttpURLConnection.HTTP_OK) error("서버 응답 ${conn.responseCode}")
                    var received = if (resumed) done else 0L
                    var lastPercent = -1L
                    conn.inputStream.use { input ->
                        FileOutputStream(part, resumed).use { output ->
                            val buffer = ByteArray(1 shl 16)
                            while (true) {
                                val n = input.read(buffer)
                                if (n < 0) break
                                output.write(buffer, 0, n)
                                received += n
                                val percent = received * 100 / m.size
                                if (percent != lastPercent) { lastPercent = percent; progress(percent.toInt()) }
                            }
                        }
                    }
                }
                if (part.length() != m.size) error("받다가 끊겼어요. 다시 누르면 이어서 받아요")
                progress(101)
                if (sha256(part) != m.sha256) {
                    part.delete()
                    error("받은 파일이 손상됐어요. 다시 받아 주세요")
                }
                if (target.exists() && target != installedModel(ctx)) target.delete()
                // 지금 쓰는 모델과 이름이 같으면 엔진을 닫고 바꾼다
                if (target.exists()) closeEngine()
                if (!part.renameTo(target)) error("파일을 저장하지 못했어요")
                val old = installedModel(ctx)
                prefs(ctx).edit().putString("file", m.file).putInt("version", m.version).apply()
                updateAvailable = false
                // 예전 모델(공식 모델이나 이전 버전)은 지워 저장 공간을 돌려준다
                if (old != null && old != target) { closeEngine(); old.delete() }
                setStatus("downloaded")
                load(ctx)
            } catch (e: Exception) {
                // 새 모델 받기에 실패해도 지금 모델이 있으면 계속 쓴다
                if (wasReady && engine != null) { updateError = e.message ?: "새 모델을 받지 못했어요"; setStatus("ready", detail) }
                else setStatus("error", e.message ?: "모델을 받지 못했어요")
            } finally {
                downloading = false
                updateProgress = -1
            }
        }.start()
    }

    private fun sha256(file: File): String {
        val md = MessageDigest.getInstance("SHA-256")
        file.inputStream().use { input ->
            val buffer = ByteArray(1 shl 20)
            while (true) {
                val n = input.read(buffer)
                if (n < 0) break
                md.update(buffer, 0, n)
            }
        }
        return md.digest().joinToString("") { "%02x".format(it) }
    }

    private fun closeEngine() = synchronized(this) {
        current?.cancel()
        conversation?.let { runCatching { it.close() } }
        conversation = null
        engine?.let { runCatching { it.close() } }
        engine = null
    }

    // ─── 메모리에 올리기 ─────────────────────────────────────────────────────────
    fun load(ctx: Context) {
        if (status != "downloaded") return
        setStatus("loading")
        loader.execute {
            val dev = devModel(ctx)
            val installed = installedModel(ctx)
            val file = dev ?: installed ?: run { setStatus("none"); return@execute }
            val path = file.absolutePath
            val version = installedVersion(ctx)
            modelName = when {
                dev != null -> "$MODEL_NAME · adb"
                version > 0 -> "$MODEL_NAME v$version"
                else -> "gemma-4-E2B 공식 (폰)"
            }
            val custom = dev != null || version > 0
            // GPU가 빠르다(공식 수치: 첫 토큰 0.3초, CPU 1.8초). GPU를 못 쓰는 폰은 CPU로
            for ((name, backend) in listOf("GPU" to Backend.GPU(), "CPU" to Backend.CPU())) {
                try {
                    val e = Engine(EngineConfig(modelPath = path, backend = backend, cacheDir = ctx.cacheDir.path))
                    e.initialize()
                    engine = e
                    setStatus("ready", if (custom) "$name · 미세조정 모델" else "$name · 공식 모델")
                    return@execute
                } catch (e: Throwable) {
                    detail = e.message ?: ""
                }
            }
            setStatus("error", "모델을 올리지 못했어요: $detail")
        }
    }

    // ─── 생성 ────────────────────────────────────────────────────────────────────
    // 이어 쓰는 대화. 매 턴 시스템 프롬프트와 지난 대화(1,000토큰 이상)를 처음부터 다시 처리하면 첫 구절까지
    // 2.4~3.2초 걸렸다(S23 FE). 지난 대화가 그대로면 대화를 이어 써서 새로 한 말만 처리한다 (서버 Ollama의 캐시와 같은 효과)
    private var conversation: Conversation? = null
    private var convSystem = ""
    private var convMessages: List<Pair<String, String>> = emptyList()  // 대화에 들어 있는 메시지 (마지막 모델 대답 제외)
    @Volatile private var convComplete = false  // 마지막 생성이 끝까지 끝났는지 (중간에 멈췄으면 상태를 믿지 않는다)

    /**
     * messages: [{"role": "system"|"user"|"assistant", "content"}] (서버가 Ollama에 보내는 것과 같다).
     * 마지막 user 메시지를 보내고, 그 앞은 대화 기록으로 넣는다. 모델을 올리는 중이면 다 올릴 때까지 기다린다.
     */
    fun start(messagesJson: String): TokenStream {
        val deadline = System.currentTimeMillis() + LOAD_TIMEOUT_MS
        while (status == "loading" && System.currentTimeMillis() < deadline) Thread.sleep(100)
        val e = engine ?: throw IllegalStateException("폰 AI가 준비되지 않았어요 ($status $detail)")

        val messages = JSONArray(messagesJson)
        var system = ""
        val turns = mutableListOf<Pair<String, String>>()
        for (i in 0 until messages.length()) {
            val m = messages.getJSONObject(i)
            if (m.getString("role") == "system") system = m.getString("content")
            else turns += m.getString("role") to m.getString("content")
        }
        val prior = turns.dropLast(1)

        synchronized(this) {
            // 지난번에 보낸 것 + 그때의 모델 대답 뒤에 새 말이 붙은 것이면 이어 쓴다.
            // (재생성처럼 마지막 말이 바뀌었거나, 생성을 중간에 멈췄으면 새로 만든다)
            val continues = conversation != null && system == convSystem &&
                prior.size == convMessages.size + 1 && prior.last().first == "assistant" &&
                prior.dropLast(1) == convMessages
            // 한 번에 한 대화만 생성한다. 앞 대답이 아직 뒤에서 생성 중이면(세 문장을 채워 화면은 이미 받았음)
            // 이어 쓸 수 있을 때는 끝날 때까지 잠깐 기다리고, 아니면 멈춘다
            current?.let { if (continues) it.awaitDone(PREVIOUS_WAIT_MS) else it.cancel() }
            val reuse = continues && convComplete
            if (!reuse) {
                conversation?.let { runCatching { it.close() } }
                conversation = e.createConversation(
                    ConversationConfig(
                        systemInstruction = Contents.of(system),
                        initialMessages = prior.map { (role, text) ->
                            if (role == "assistant") Message.model(text) else Message.user(text)
                        },
                        samplerConfig = SamplerConfig(40, 0.95, TEMPERATURE, 0),
                        maxOutputToken = MAX_OUTPUT_TOKENS,
                        // Gemma 4는 기본으로 영어로 길게 "생각"한 뒤 답한다 (첫 인사에서 343토큰 생각 → 17토큰 답).
                        // 생각을 켜 두면 첫 소리가 늦고 출력 한도(120토큰)를 다 써서 답이 비기도 한다
                        thinkingConfig = ThinkingConfig(enableThinking = false),
                    ),
                )
                convSystem = system
            }
            convMessages = turns
            convComplete = false
            val stream = TokenStream(conversation!!) { convComplete = true }
            current = stream
            conversation!!.sendMessageAsync(turns.last().second, stream.callback)
            return stream
        }
    }

    /** 화면이 대답을 멈췄을 때 (사용자가 끼어듦) */
    fun cancelCurrent() = synchronized(this) { current?.cancel() }

    /** Python(engine/phone.py)이 next()가 null을 줄 때까지 토큰을 읽는다 */
    class TokenStream(private val conversation: Conversation, private val onComplete: () -> Unit) {
        private val queue = LinkedBlockingQueue<Any>()
        private object End
        @Volatile private var closed = false
        @Volatile private var cancelled = false
        private val finished = java.util.concurrent.CountDownLatch(1)

        val callback = object : MessageCallback {
            override fun onMessage(message: Message) {
                val text = message.contents.contents.filterIsInstance<Content.Text>().joinToString("") { it.text }
                if (text.isNotEmpty() && !closed) queue.put(text)
            }
            override fun onDone() {
                if (!cancelled) onComplete()
                finished.countDown()
                queue.put(End)
            }
            override fun onError(throwable: Throwable) {
                finished.countDown()
                queue.put(throwable)
            }
        }

        fun next(): String? {
            if (closed) return null
            return when (val item = queue.poll(TOKEN_TIMEOUT_S, TimeUnit.SECONDS)) {
                null -> throw IllegalStateException("폰 AI 응답이 ${TOKEN_TIMEOUT_S}초 동안 없어요")
                End -> null
                is Throwable -> throw IllegalStateException(item.message ?: "생성 실패", item)
                else -> item as String
            }
        }

        /**
         * 대화 엔진이 더 읽지 않을 때(세 문장을 채움 등). 남은 생성은 뒤에서 끝까지 마치게 둔다.
         * 중간에 멈추면 대화 상태를 믿을 수 없어 다음 턴에 처음부터 다시 처리해야 한다(첫 구절 0.7초 → 2.7초)
         */
        fun close() {
            if (closed) return
            closed = true
            queue.put(End)
        }

        /** 사용자가 끼어들었거나 재생성할 때: 바로 멈춘다 (이 대화는 다음에 이어 쓰지 않는다) */
        fun cancel() {
            close()
            if (finished.count > 0) {
                cancelled = true
                runCatching { conversation.cancelProcess() }
            }
        }

        fun awaitDone(ms: Long) {
            if (!finished.await(ms, TimeUnit.MILLISECONDS)) cancel()
        }
    }
}
