package com.morningcall.app

import android.Manifest
import android.annotation.SuppressLint
import android.content.BroadcastReceiver
import android.content.Context
import android.content.Intent
import android.content.IntentFilter
import android.content.pm.PackageManager
import android.hardware.Sensor
import android.hardware.SensorEvent
import android.hardware.SensorEventListener
import android.hardware.SensorManager
import android.os.Build
import android.os.Handler
import android.os.Looper
import android.os.PowerManager
import org.json.JSONArray
import org.json.JSONObject

/**
 * 통화가 끝난 뒤 정해진 시간(기본 5분) 동안 정말 일어났는지 볼 수 있는 행동을 잰다.
 * 걸음 수(걸음 센서), 화면이 켜져 있던 시간·꺼진 횟수, 잠금 해제, 다른 앱으로 넘어간 횟수.
 * 앱 화면이 닫혀도 계속 재도록 앱 Context에 붙이고, 결과는 저장해 두었다가 웹 화면이 가져가 서버로 보낸다.
 */
object WakeMonitor : SensorEventListener {
    private const val KEY_REPORTS = "pending_wake_reports"
    private const val MAX_REPORTS = 10
    private const val FLUSH_WAIT_MS = 1_000L  // 걸음 센서가 모아 둔 값을 내보낼 때까지 기다리는 시간

    private val handler = Handler(Looper.getMainLooper())
    @SuppressLint("StaticFieldLeak")  // 앱 Context(applicationContext)만 담으므로 화면이 새지 않는다
    private var app: Context? = null
    private var wakeLock: PowerManager.WakeLock? = null

    private var sessionId = ""
    private var windowMs = 0L
    private var startedAt = 0L
    private var hasStepSensor = false
    private var stepBase = -1f
    private var stepLast = -1f
    private var screenOn = true
    private var screenOnSince = 0L
    private var screenOnMs = 0L
    private var screenOffCount = 0
    private var firstScreenOffMs = -1L
    private var unlockCount = 0
    private var appVisible = true
    private var appVisibleSince = 0L
    private var foregroundMs = 0L
    private var appLeftCount = 0
    private var firstAppLeftMs = -1L

    /** 결과가 저장되면 불린다. 웹 화면이 떠 있으면 바로 가져가게 MainActivity가 연결한다 */
    var onReport: (() -> Unit)? = null

    private val finishRunnable = Runnable { finish() }

    private val screenReceiver = object : BroadcastReceiver() {
        override fun onReceive(context: Context, intent: Intent) {
            val now = System.currentTimeMillis()
            when (intent.action) {
                Intent.ACTION_SCREEN_OFF -> if (screenOn) {
                    screenOn = false
                    screenOnMs += now - screenOnSince
                    screenOffCount += 1
                    if (firstScreenOffMs < 0) firstScreenOffMs = now - startedAt
                }
                Intent.ACTION_SCREEN_ON -> if (!screenOn) {
                    screenOn = true
                    screenOnSince = now
                }
                Intent.ACTION_USER_PRESENT -> unlockCount += 1
            }
        }
    }

    fun start(context: Context, session: String, seconds: Int) {
        if (app != null) finish()  // 앞의 측정이 남아 있으면 지금까지 잰 걸로 마무리
        val ctx = context.applicationContext
        app = ctx
        val now = System.currentTimeMillis()
        sessionId = session
        windowMs = seconds * 1000L
        startedAt = now
        stepBase = -1f
        stepLast = -1f
        screenOn = ctx.getSystemService(PowerManager::class.java).isInteractive
        screenOnSince = now
        screenOnMs = 0L
        screenOffCount = 0
        firstScreenOffMs = -1L
        unlockCount = 0
        appVisible = true  // 통화를 끝낸 직후라 앱 화면이 떠 있다
        appVisibleSince = now
        foregroundMs = 0L
        appLeftCount = 0
        firstAppLeftMs = -1L

        ctx.registerReceiver(screenReceiver, IntentFilter().apply {
            addAction(Intent.ACTION_SCREEN_ON)
            addAction(Intent.ACTION_SCREEN_OFF)
            addAction(Intent.ACTION_USER_PRESENT)
        })
        val sensors = ctx.getSystemService(SensorManager::class.java)
        val stepCounter = sensors.getDefaultSensor(Sensor.TYPE_STEP_COUNTER)
        hasStepSensor = stepCounter != null && canCountSteps(ctx) &&
            sensors.registerListener(this, stepCounter, SensorManager.SENSOR_DELAY_NORMAL)

        // 화면이 꺼지면 CPU가 잠들어 타이머가 밀리므로, 측정하는 동안만 CPU를 깨워 둔다
        wakeLock = ctx.getSystemService(PowerManager::class.java)
            .newWakeLock(PowerManager.PARTIAL_WAKE_LOCK, "MorningCall:WakeMonitor")
            .apply { acquire(windowMs + FLUSH_WAIT_MS + 10_000L) }
        handler.postDelayed(finishRunnable, windowMs)
    }

    /** MainActivity의 onStart/onStop. 화면이 켜진 채로 앱을 벗어나면 다른 앱으로 넘어간 것으로 본다 */
    fun onAppVisible(context: Context, visible: Boolean) {
        if (app == null || visible == appVisible) return
        val now = System.currentTimeMillis()
        appVisible = visible
        if (visible) {
            appVisibleSince = now
            return
        }
        foregroundMs += now - appVisibleSince
        // 화면을 끄면 onStop도 불리는데, 이때는 이미 isInteractive가 false다
        if (context.getSystemService(PowerManager::class.java).isInteractive) {
            appLeftCount += 1
            if (firstAppLeftMs < 0) firstAppLeftMs = now - startedAt
        }
    }

    /** 저장된 결과(JSON 배열 문자열)를 꺼내고 지운다 */
    fun takeReports(context: Context): String {
        val prefs = context.prefs()
        val reports = prefs.getString(KEY_REPORTS, null) ?: return "[]"
        prefs.edit().remove(KEY_REPORTS).apply()
        return reports
    }

    override fun onSensorChanged(event: SensorEvent) {
        // 걸음 센서는 부팅 후 누적 걸음 수를 주므로, 처음 받은 값과의 차이가 측정 중 걸음 수다
        val total = event.values[0]
        if (stepBase < 0) stepBase = total
        stepLast = total
    }

    override fun onAccuracyChanged(sensor: Sensor?, accuracy: Int) {}

    private fun finish() {
        val ctx = app ?: return
        handler.removeCallbacks(finishRunnable)
        // 걸음 센서는 값을 모아 두었다가 늦게 보낼 수 있으므로, 남은 값을 받고 마무리한다
        val sensors = ctx.getSystemService(SensorManager::class.java)
        if (hasStepSensor) sensors.flush(this)
        val report = buildReport()
        app = null
        handler.postDelayed({
            report.put("steps", if (hasStepSensor) stepCount() else JSONObject.NULL)
            sensors.unregisterListener(this)
            try { ctx.unregisterReceiver(screenReceiver) } catch (_: IllegalArgumentException) {}
            wakeLock?.takeIf { it.isHeld }?.release()
            wakeLock = null
            save(ctx, report)
            onReport?.invoke()
        }, FLUSH_WAIT_MS)
    }

    private fun buildReport(): JSONObject {
        val now = System.currentTimeMillis()
        val elapsed = now - startedAt
        val onMs = screenOnMs + if (screenOn) now - screenOnSince else 0L
        val fgMs = foregroundMs + if (appVisible) now - appVisibleSince else 0L
        return JSONObject().apply {
            put("session_id", sessionId)
            put("window_sec", seconds(elapsed))
            put("complete", elapsed >= windowMs)
            put("step_sensor", hasStepSensor)
            put("screen_on_sec", seconds(onMs))
            put("screen_off_count", screenOffCount)
            put("first_screen_off_sec", if (firstScreenOffMs < 0) JSONObject.NULL else seconds(firstScreenOffMs))
            put("unlock_count", unlockCount)
            put("foreground_sec", seconds(fgMs))
            put("app_left_count", appLeftCount)
            put("first_app_left_sec", if (firstAppLeftMs < 0) JSONObject.NULL else seconds(firstAppLeftMs))
        }
    }

    private fun stepCount(): Int = if (stepBase < 0) 0 else (stepLast - stepBase).toInt()

    private fun seconds(ms: Long): Double = Math.round(ms / 100.0) / 10.0

    private fun canCountSteps(context: Context) = Build.VERSION.SDK_INT < 29 ||
        context.checkSelfPermission(Manifest.permission.ACTIVITY_RECOGNITION) == PackageManager.PERMISSION_GRANTED

    private fun save(context: Context, report: JSONObject) {
        val prefs = context.prefs()
        val reports = try { JSONArray(prefs.getString(KEY_REPORTS, "[]")) } catch (_: Exception) { JSONArray() }
        reports.put(report)
        // 웹 화면이 오래 가져가지 않으면 오래된 것부터 버린다
        while (reports.length() > MAX_REPORTS) reports.remove(0)
        prefs.edit().putString(KEY_REPORTS, reports.toString()).apply()
    }
}
