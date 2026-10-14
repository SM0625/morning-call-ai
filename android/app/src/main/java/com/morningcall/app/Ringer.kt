package com.morningcall.app

import android.content.Context
import android.media.AudioAttributes
import android.media.MediaPlayer
import android.media.RingtoneManager
import android.os.VibrationEffect
import android.os.Vibrator

/** 전화 오는 소리. 무음 모드에서도 들리도록 알람 음량 채널로 재생한다. */
object Ringer {
    private var player: MediaPlayer? = null
    private var vibrator: Vibrator? = null

    @Synchronized
    fun start(context: Context) {
        if (player != null) return
        val app = context.applicationContext
        val uri = RingtoneManager.getDefaultUri(RingtoneManager.TYPE_ALARM)
            ?: RingtoneManager.getDefaultUri(RingtoneManager.TYPE_RINGTONE)
        player = try {
            MediaPlayer().apply {
                setAudioAttributes(
                    AudioAttributes.Builder()
                        .setUsage(AudioAttributes.USAGE_ALARM)
                        .setContentType(AudioAttributes.CONTENT_TYPE_SONIFICATION)
                        .build(),
                )
                setDataSource(app, uri)
                isLooping = true
                prepare()
                start()
            }
        } catch (e: Exception) {
            null
        }
        @Suppress("DEPRECATION")
        vibrator = app.getSystemService(Vibrator::class.java)?.also {
            it.vibrate(VibrationEffect.createWaveform(longArrayOf(0, 800, 600), 0))
        }
    }

    @Synchronized
    fun stop(context: Context) {
        player?.run {
            try { stop() } catch (_: Exception) {}
            release()
        }
        player = null
        vibrator?.cancel()
        vibrator = null
        AlarmNotification.cancel(context)
    }
}
