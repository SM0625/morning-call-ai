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
import java.io.File
import java.io.FileOutputStream
import java.net.HttpURLConnection
import java.net.URL
import java.util.concurrent.Executors
import java.util.concurrent.LinkedBlockingQueue
import java.util.concurrent.TimeUnit

/**
 * 폰 안에서 Gemma 4 E2B를 돌린다 (Google LiteRT-LM). PC 서버·와이파이가 꺼져 있어도 깨울 수 있게 하려는 것.
 * 모델(약 2.6GB)은 처음 한 번 Hugging Face에서 받아 앱 저장소에 둔다.
 * 대화 엔진(engine/dialog.py)이 [start]로 토큰 스트림을 받아 쓴다 (engine/phone.py).
 */
object PhoneAi {
    const val MODEL_NAME = "gemma-4-E2B (폰)"
    // Apache 2.0 라이선스라 로그인·동의 없이 받을 수 있다
    private const val MODEL_URL =
        "https://huggingface.co/litert-community/gemma-4-E2B-it-litert-lm/resolve/main/gemma-4-E2B-it.litertlm"
    // 서버(main.py GENERATION_OPTIONS)와 같은 생성 설정
    private const val TEMPERATURE = 0.7
    private const val MAX_OUTPUT_TOKENS = 120
    private const val TOKEN_TIMEOUT_S = 60L
    private const val LOAD_TIMEOUT_MS = 90_000L
    // 앞 대답의 남은 생성(최대 120토큰, 약 2초)을 기다리는 시간. 보통은 AI가 말하는 동안 이미 끝나 있다
    private const val PREVIOUS_WAIT_MS = 3_000L

    /** none(모델 없음) | downloading | downloaded(받았지만 아직 안 올림) | loading | ready | error */
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

    private fun modelFile(ctx: Context) = File(ctx.filesDir, "models/gemma-4-E2B-it.litertlm")

    /** 앱을 열 때: 받아 둔 모델이 있으면 바로 메모리에 올린다 (알람이 울릴 때 기다리지 않도록) */
    fun init(ctx: Context) {
        if (status == "none" && modelFile(ctx).exists()) setStatus("downloaded")
        load(ctx)
    }

    // ─── 모델 받기 ───────────────────────────────────────────────────────────────
    /** 이어 받기를 지원한다. 끊겨도 다시 누르면 받은 곳부터 이어서 받는다 */
    fun download(ctx: Context) {
        if (status != "none" && status != "error") return
        val target = modelFile(ctx)
        if (target.exists()) { setStatus("downloaded"); load(ctx); return }
        setStatus("downloading", "0")
        Thread {
            try {
                target.parentFile?.mkdirs()
                val part = File(target.path + ".part")
                val done = if (part.exists()) part.length() else 0L
                val conn = (URL(MODEL_URL).openConnection() as HttpURLConnection).apply {
                    connectTimeout = 15_000
                    readTimeout = 30_000
                    if (done > 0) setRequestProperty("Range", "bytes=$done-")
                }
                val resumed = conn.responseCode == HttpURLConnection.HTTP_PARTIAL
                if (!resumed && conn.responseCode != HttpURLConnection.HTTP_OK) error("서버 응답 ${conn.responseCode}")
                val start = if (resumed) done else 0L
                val total = start + conn.contentLengthLong
                if (ctx.filesDir.usableSpace < total - start + 200_000_000L) error("저장 공간이 부족해요 (약 2.6GB 필요)")
                var received = start
                var lastPercent = -1L
                conn.inputStream.use { input ->
                    FileOutputStream(part, resumed).use { output ->
                        val buffer = ByteArray(1 shl 16)
                        while (true) {
                            val n = input.read(buffer)
                            if (n < 0) break
                            output.write(buffer, 0, n)
                            received += n
                            val percent = received * 100 / total
                            if (percent != lastPercent) { lastPercent = percent; setStatus("downloading", "$percent") }
                        }
                    }
                }
                if (received < total) error("받다가 끊겼어요. 다시 누르면 이어서 받아요")
                if (!part.renameTo(target)) error("파일을 저장하지 못했어요")
                setStatus("downloaded")
                load(ctx)
            } catch (e: Exception) {
                setStatus("error", e.message ?: "모델을 받지 못했어요")
            }
        }.start()
    }

    // ─── 메모리에 올리기 ─────────────────────────────────────────────────────────
    fun load(ctx: Context) {
        if (status != "downloaded") return
        setStatus("loading")
        loader.execute {
            val path = modelFile(ctx).absolutePath
            // GPU가 빠르다(공식 수치: 첫 토큰 0.3초, CPU 1.8초). GPU를 못 쓰는 폰은 CPU로
            for ((name, backend) in listOf("GPU" to Backend.GPU(), "CPU" to Backend.CPU())) {
                try {
                    val e = Engine(EngineConfig(modelPath = path, backend = backend, cacheDir = ctx.cacheDir.path))
                    e.initialize()
                    engine = e
                    setStatus("ready", name)
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
