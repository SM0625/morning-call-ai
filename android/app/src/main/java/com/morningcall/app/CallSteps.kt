package com.morningcall.app

import android.Manifest
import android.content.Context
import android.content.pm.PackageManager
import android.hardware.Sensor
import android.hardware.SensorEvent
import android.hardware.SensorEventListener
import android.hardware.SensorManager
import android.os.Build

/**
 * 통화 중 걸음 수. 목표 걸음 수를 채우면 통화를 끝내는 데 쓴다(화면이 목표와 비교한다).
 * 걸음마다 바로 오는 걸음 감지 센서를 먼저 쓰고, 없으면 누적 걸음 센서의 차이로 센다.
 * 통화 후 5분 측정(WakeMonitor)과는 켜져 있는 시간이 달라 따로 둔다.
 */
class CallSteps(
    private val context: Context,
    private val onSteps: (Int) -> Unit,
) : SensorEventListener {
    private val sensors = context.getSystemService(SensorManager::class.java)
    private var steps = 0
    private var counterBase = -1f
    private var running = false

    /** @return 시작하지 못한 이유("permission", "sensor"). 시작했으면 null */
    fun start(): String? {
        stop()
        if (Build.VERSION.SDK_INT >= 29 &&
            context.checkSelfPermission(Manifest.permission.ACTIVITY_RECOGNITION) != PackageManager.PERMISSION_GRANTED
        ) return "permission"
        steps = 0
        counterBase = -1f
        // 걸음 감지 센서는 걸음마다 바로 알려 준다. 누적 센서는 값을 모았다가 늦게 보내는 기기가 있다
        val sensor = sensors.getDefaultSensor(Sensor.TYPE_STEP_DETECTOR)
            ?: sensors.getDefaultSensor(Sensor.TYPE_STEP_COUNTER)
            ?: return "sensor"
        running = sensors.registerListener(this, sensor, SensorManager.SENSOR_DELAY_FASTEST)
        if (!running) return "sensor"
        onSteps(0)
        return null
    }

    fun stop() {
        if (running) sensors.unregisterListener(this)
        running = false
    }

    override fun onSensorChanged(event: SensorEvent) {
        when (event.sensor.type) {
            Sensor.TYPE_STEP_DETECTOR -> steps += 1
            Sensor.TYPE_STEP_COUNTER -> {
                // 부팅 후 누적 걸음 수라, 처음 받은 값과의 차이가 통화 중 걸음 수다
                val total = event.values[0]
                if (counterBase < 0) counterBase = total
                steps = (total - counterBase).toInt()
            }
        }
        onSteps(steps)
    }

    override fun onAccuracyChanged(sensor: Sensor?, accuracy: Int) {}
}
