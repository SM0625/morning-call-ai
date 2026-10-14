package com.morningcall.app

import android.Manifest
import android.app.Activity
import android.app.NotificationManager
import android.content.Intent
import android.content.pm.PackageManager
import android.media.AudioAttributes
import android.net.Uri
import android.os.Build
import android.os.Bundle
import android.os.Handler
import android.os.Looper
import android.provider.Settings
import android.speech.RecognitionListener
import android.speech.RecognizerIntent
import android.speech.SpeechRecognizer
import android.speech.tts.TextToSpeech
import android.speech.tts.UtteranceProgressListener
import android.view.WindowManager
import android.webkit.GeolocationPermissions
import android.webkit.JavascriptInterface
import android.webkit.WebChromeClient
import android.webkit.WebView
import android.webkit.WebViewClient
import android.widget.Toast
import org.json.JSONObject
import java.text.SimpleDateFormat
import java.util.Date
import java.util.Locale

/**
 * 웹 화면(static/index.html)을 띄우고, 웹에서 할 수 없는 일(알람, 벨소리, 음성 합성/인식)을
 * MorningCallApp 브리지로 제공한다. 안드로이드 WebView는 Web Speech API를 지원하지 않는다.
 */
class MainActivity : Activity() {

    companion object {
        const val EXTRA_ALARM = "alarm"
        private const val KEY_SERVER_URL = "server_url"
        private const val REQ_PERMISSIONS = 10
    }

    private lateinit var web: WebView
    private var tts: TextToSpeech? = null
    private var ttsState = 0  // 0: 초기화 중, 1: 사용 가능, -1: 사용 불가
    private val pendingSpeech = mutableListOf<Pair<String, String>>()  // (문장, id) — TTS 초기화 전에 들어온 요청
    private var recognizer: SpeechRecognizer? = null
    private val handler = Handler(Looper.getMainLooper())
    private var lastPartial = ""
    // 중간 인식 결과가 이 시간 동안 바뀌지 않으면 말이 끝난 것으로 보고 바로 인식을 마친다.
    // 기본 인식기는 1~2초 침묵을 기다리는데, 사람 사이 말 차례 간격은 보통 1초 이내다.
    private val endOfSpeechMs = 800L
    private val finishListening = Runnable { recognizer?.stopListening() }
    private var pageLoaded = false
    private var pendingAlarm = false

    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        window.addFlags(WindowManager.LayoutParams.FLAG_KEEP_SCREEN_ON)

        web = WebView(this)
        setContentView(web)
        setupWebView()
        setupTts()
        consumeAlarmExtra(intent)
        requestNeededPermissions()
        web.loadUrl("file:///android_asset/index.html")
    }

    override fun onNewIntent(intent: Intent) {
        super.onNewIntent(intent)
        setIntent(intent)
        consumeAlarmExtra(intent)
    }

    override fun onDestroy() {
        tts?.shutdown()
        recognizer?.destroy()
        if (isFinishing) Ringer.stop(this)
        web.destroy()
        super.onDestroy()
    }

    // ─── 알람으로 열렸는지 ───────────────────────────────────────────────────────
    private fun consumeAlarmExtra(intent: Intent) {
        if (!intent.getBooleanExtra(EXTRA_ALARM, false)) return
        // 화면 회전 등으로 다시 만들어질 때 알람이 또 울리지 않도록 한 번 쓰고 지운다
        intent.removeExtra(EXTRA_ALARM)
        if (pageLoaded) fireAlarm() else pendingAlarm = true
    }

    private fun fireAlarm() {
        pendingAlarm = false
        js("window.onNativeAlarm && window.onNativeAlarm()")
    }

    // ─── WebView ────────────────────────────────────────────────────────────────
    private fun setupWebView() {
        with(web.settings) {
            javaScriptEnabled = true
            domStorageEnabled = true  // localStorage에 이름·일정 저장
            mediaPlaybackRequiresUserGesture = false
            allowFileAccess = true
            // file:// 로 연 화면에서 PC 서버(http://...)로 요청을 보내기 위해 필요
            @Suppress("DEPRECATION")
            allowUniversalAccessFromFileURLs = true
            setGeolocationEnabled(true)
        }
        web.webViewClient = object : WebViewClient() {
            override fun onPageFinished(view: WebView, url: String) {
                pageLoaded = true
                if (pendingAlarm) fireAlarm()
            }
        }
        web.webChromeClient = object : WebChromeClient() {
            override fun onGeolocationPermissionsShowPrompt(
                origin: String?, callback: GeolocationPermissions.Callback?,
            ) {
                callback?.invoke(origin, hasPermission(Manifest.permission.ACCESS_COARSE_LOCATION), false)
            }
        }
        web.addJavascriptInterface(Bridge(), "MorningCallApp")
    }

    private fun js(code: String) = runOnUiThread { web.evaluateJavascript(code, null) }

    // ─── 말하기 (TTS) ────────────────────────────────────────────────────────────
    private fun setupTts() {
        tts = TextToSpeech(this) { status ->
            val engine = tts
            if (status == TextToSpeech.SUCCESS && engine != null) {
                val lang = engine.setLanguage(Locale.KOREAN)
                ttsState = if (lang >= TextToSpeech.LANG_AVAILABLE) 1 else -1
                engine.setSpeechRate(1.05f)
                engine.setPitch(1.1f)
                // 미디어 음량이 0이어도 들리도록 알람 채널로 말한다
                engine.setAudioAttributes(
                    AudioAttributes.Builder()
                        .setUsage(AudioAttributes.USAGE_ALARM)
                        .setContentType(AudioAttributes.CONTENT_TYPE_SPEECH)
                        .build(),
                )
                engine.setOnUtteranceProgressListener(object : UtteranceProgressListener() {
                    override fun onStart(utteranceId: String?) {}
                    override fun onDone(utteranceId: String?) = speakDone(utteranceId)
                    @Deprecated("Deprecated in Java")
                    override fun onError(utteranceId: String?) = speakDone(utteranceId)
                })
            } else {
                ttsState = -1
            }
            if (ttsState == -1) {
                runOnUiThread { Toast.makeText(this, "한국어 음성 합성을 쓸 수 없어요. 설정에서 TTS 엔진을 확인해 주세요.", Toast.LENGTH_LONG).show() }
            }
            // 알람으로 앱이 막 켜졌을 때는 초기화 전에 말하기 요청이 올 수 있다
            runOnUiThread {
                val queued = pendingSpeech.toList()
                pendingSpeech.clear()
                queued.forEach { (text, id) -> enqueueSpeech(text, id) }
            }
        }
    }

    private fun speakDone(id: String?) = js("window.onNativeSpeakDone(${JSONObject.quote(id ?: "")})")

    /** 대기열 뒤에 붙인다. 앞 구절을 읽는 동안 다음 구절을 미리 합성해서 끊김 없이 이어 읽는다. */
    private fun enqueueSpeech(text: String, id: String) {
        when (ttsState) {
            0 -> pendingSpeech += text to id
            1 -> tts?.speak(text, TextToSpeech.QUEUE_ADD, null, id)
            else -> speakDone(id)  // 말할 수 없으면 바로 다음으로 넘어간다
        }
    }

    // ─── 듣기 (STT) ──────────────────────────────────────────────────────────────
    private val recognitionListener = object : RecognitionListener {
        override fun onResults(results: Bundle) {
            handler.removeCallbacks(finishListening)
            val text = results.getStringArrayList(SpeechRecognizer.RESULTS_RECOGNITION)?.firstOrNull()
                ?.takeIf { it.isNotBlank() } ?: lastPartial.takeIf { it.isNotBlank() }
            if (text == null) speechError("no-speech")
            else js("window.onNativeSpeechResult(${JSONObject.quote(text)})")
        }

        override fun onPartialResults(partialResults: Bundle) {
            val text = partialResults.getStringArrayList(SpeechRecognizer.RESULTS_RECOGNITION)?.firstOrNull()
            if (!text.isNullOrBlank() && text != lastPartial) {
                lastPartial = text
                handler.removeCallbacks(finishListening)
                handler.postDelayed(finishListening, endOfSpeechMs)
            }
        }

        override fun onError(error: Int) {
            handler.removeCallbacks(finishListening)
            // 일찍 끊어서 최종 결과가 안 나와도 중간 결과가 있으면 그걸 쓴다
            if (lastPartial.isNotBlank() && error == SpeechRecognizer.ERROR_NO_MATCH) {
                js("window.onNativeSpeechResult(${JSONObject.quote(lastPartial)})")
                return
            }
            speechError(
                when (error) {
                    SpeechRecognizer.ERROR_SPEECH_TIMEOUT, SpeechRecognizer.ERROR_NO_MATCH -> "no-speech"
                    SpeechRecognizer.ERROR_INSUFFICIENT_PERMISSIONS -> "not-allowed"
                    SpeechRecognizer.ERROR_NETWORK, SpeechRecognizer.ERROR_NETWORK_TIMEOUT,
                    SpeechRecognizer.ERROR_SERVER -> "network"
                    else -> "error"
                },
            )
        }

        override fun onReadyForSpeech(params: Bundle?) {}
        override fun onBeginningOfSpeech() {}
        override fun onRmsChanged(rmsdB: Float) {}
        override fun onBufferReceived(buffer: ByteArray?) {}
        override fun onEndOfSpeech() {}
        override fun onEvent(eventType: Int, params: Bundle?) {}
    }

    private fun speechError(code: String) = js("window.onNativeSpeechError('$code')")

    private fun listen() {
        if (!hasPermission(Manifest.permission.RECORD_AUDIO)) {
            speechError("not-allowed")
            requestPermissions(arrayOf(Manifest.permission.RECORD_AUDIO), REQ_PERMISSIONS)
            return
        }
        if (!SpeechRecognizer.isRecognitionAvailable(this)) {
            speechError("error")
            return
        }
        val engine = recognizer ?: SpeechRecognizer.createSpeechRecognizer(this).also {
            it.setRecognitionListener(recognitionListener)
            recognizer = it
        }
        engine.cancel()
        handler.removeCallbacks(finishListening)
        lastPartial = ""
        engine.startListening(
            Intent(RecognizerIntent.ACTION_RECOGNIZE_SPEECH).apply {
                putExtra(RecognizerIntent.EXTRA_LANGUAGE_MODEL, RecognizerIntent.LANGUAGE_MODEL_FREE_FORM)
                putExtra(RecognizerIntent.EXTRA_LANGUAGE, "ko-KR")
                putExtra(RecognizerIntent.EXTRA_MAX_RESULTS, 1)
                putExtra(RecognizerIntent.EXTRA_PARTIAL_RESULTS, true)
                // 인식기가 지원하면 말 끝 침묵 대기 시간도 줄인다 (지원하지 않으면 무시됨)
                putExtra(RecognizerIntent.EXTRA_SPEECH_INPUT_COMPLETE_SILENCE_LENGTH_MILLIS, endOfSpeechMs)
                putExtra(RecognizerIntent.EXTRA_SPEECH_INPUT_POSSIBLY_COMPLETE_SILENCE_LENGTH_MILLIS, endOfSpeechMs)
            },
        )
    }

    private fun stopListeningNative() {
        handler.removeCallbacks(finishListening)
        recognizer?.cancel()
    }

    // ─── 알람 ────────────────────────────────────────────────────────────────────
    private fun scheduleAlarm(triggerAt: Long) {
        val exact = AlarmScheduler.schedule(this, triggerAt)
        val time = SimpleDateFormat("M월 d일 HH:mm", Locale.KOREA).format(Date(triggerAt))
        val message = if (exact) "$time 에 모닝콜이 울려요"
        else "$time 알람을 등록했지만, 정확한 알람 권한이 없어 조금 늦게 울릴 수 있어요"
        runOnUiThread { Toast.makeText(this, message, Toast.LENGTH_LONG).show() }
    }

    // ─── 권한 ────────────────────────────────────────────────────────────────────
    private fun hasPermission(permission: String) =
        checkSelfPermission(permission) == PackageManager.PERMISSION_GRANTED

    private fun requestNeededPermissions() {
        val needed = mutableListOf(Manifest.permission.RECORD_AUDIO, Manifest.permission.ACCESS_COARSE_LOCATION)
        if (Build.VERSION.SDK_INT >= 33) needed += Manifest.permission.POST_NOTIFICATIONS
        val missing = needed.filterNot { hasPermission(it) }
        if (missing.isNotEmpty()) requestPermissions(missing.toTypedArray(), REQ_PERMISSIONS)

        // Android 14부터는 잠금화면 위 전체 화면 알림 권한을 따로 켜야 할 수 있다
        if (Build.VERSION.SDK_INT >= 34 &&
            !getSystemService(NotificationManager::class.java).canUseFullScreenIntent()
        ) {
            Toast.makeText(this, "알람이 잠금화면 위에 뜨도록 '전체 화면 알림'을 허용해 주세요", Toast.LENGTH_LONG).show()
            startActivity(
                Intent(Settings.ACTION_MANAGE_APP_USE_FULL_SCREEN_INTENT, Uri.parse("package:$packageName")),
            )
        }
    }

    // ─── 웹 화면에서 호출하는 브리지 ──────────────────────────────────────────────
    inner class Bridge {
        /** 직접 입력한 주소가 없으면 빌드할 때 넣어 둔 PC 주소 */
        @JavascriptInterface
        fun getServerUrl(): String =
            prefs().getString(KEY_SERVER_URL, null)?.takeIf { it.isNotBlank() } ?: BuildConfig.DEFAULT_SERVER_URL

        @JavascriptInterface
        fun setServerUrl(url: String) {
            prefs().edit().putString(KEY_SERVER_URL, url.trim()).apply()
        }

        /** 말하기 대기열에 추가. 다 읽으면 window.onNativeSpeakDone(id) */
        @JavascriptInterface
        fun speak(text: String, id: String) = runOnUiThread { enqueueSpeech(text, id) }

        @JavascriptInterface
        fun stopSpeaking() = runOnUiThread { pendingSpeech.clear(); tts?.stop() }

        @JavascriptInterface
        fun startListening() = runOnUiThread { listen() }

        @JavascriptInterface
        fun stopListening() = runOnUiThread { stopListeningNative() }

        @JavascriptInterface
        fun setAlarm(hour: Int, minute: Int) = scheduleAlarm(AlarmScheduler.nextTrigger(hour, minute))

        @JavascriptInterface
        fun snooze(minutes: Int) = scheduleAlarm(System.currentTimeMillis() + minutes * 60_000L)

        @JavascriptInterface
        fun cancelAlarm() = AlarmScheduler.cancel(this@MainActivity)

        @JavascriptInterface
        fun startRinging() = Ringer.start(this@MainActivity)

        @JavascriptInterface
        fun stopRinging() = Ringer.stop(this@MainActivity)
    }
}
