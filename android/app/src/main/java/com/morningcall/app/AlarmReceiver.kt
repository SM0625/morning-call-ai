package com.morningcall.app

import android.app.Notification
import android.app.NotificationChannel
import android.app.NotificationManager
import android.app.PendingIntent
import android.content.BroadcastReceiver
import android.content.Context
import android.content.Intent

/** 알람 시각에 실행된다. 벨을 울리고, 잠금화면 위에 전화 화면(MainActivity)을 띄운다. */
class AlarmReceiver : BroadcastReceiver() {
    override fun onReceive(context: Context, intent: Intent) {
        AlarmScheduler.clearSaved(context)
        Ringer.start(context)
        AlarmNotification.show(context)
        // 앱이 이미 화면에 떠 있으면 바로 전화 화면으로 전환 (백그라운드에서는 시스템이 막으므로 알림이 대신 띄운다)
        try {
            context.startActivity(AlarmNotification.launchIntent(context))
        } catch (_: Exception) {
        }
    }
}

/** 화면이 꺼져 있거나 잠겨 있을 때 전화처럼 전체 화면으로 뜨는 알림 */
object AlarmNotification {
    private const val CHANNEL_ID = "morning_call"
    private const val NOTIFICATION_ID = 1
    private const val REQ_LAUNCH = 102

    fun launchIntent(context: Context): Intent = Intent(context, MainActivity::class.java).apply {
        putExtra(MainActivity.EXTRA_ALARM, true)
        addFlags(Intent.FLAG_ACTIVITY_NEW_TASK or Intent.FLAG_ACTIVITY_SINGLE_TOP)
    }

    fun show(context: Context) {
        val manager = context.getSystemService(NotificationManager::class.java)
        manager.createNotificationChannel(
            NotificationChannel(CHANNEL_ID, "모닝콜 전화", NotificationManager.IMPORTANCE_HIGH).apply {
                description = "알람 시각에 모닝콜 전화 화면을 띄웁니다"
                setSound(null, null)  // 소리는 Ringer가 낸다
                lockscreenVisibility = Notification.VISIBILITY_PUBLIC
            },
        )
        val pending = PendingIntent.getActivity(
            context, REQ_LAUNCH, launchIntent(context),
            PendingIntent.FLAG_UPDATE_CURRENT or PendingIntent.FLAG_IMMUTABLE,
        )
        val notification = Notification.Builder(context, CHANNEL_ID)
            .setSmallIcon(R.drawable.ic_alarm)
            .setContentTitle("AI 모닝콜")
            .setContentText("전화 오는 중… 받아서 깨어나세요!")
            .setCategory(Notification.CATEGORY_CALL)
            .setVisibility(Notification.VISIBILITY_PUBLIC)
            .setFullScreenIntent(pending, true)
            .setContentIntent(pending)
            .setOngoing(true)
            .build()
        manager.notify(NOTIFICATION_ID, notification)
    }

    fun cancel(context: Context) {
        context.getSystemService(NotificationManager::class.java).cancel(NOTIFICATION_ID)
    }
}
