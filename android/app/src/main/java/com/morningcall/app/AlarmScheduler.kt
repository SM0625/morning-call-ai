package com.morningcall.app

import android.app.AlarmManager
import android.app.PendingIntent
import android.content.Context
import android.content.Intent
import android.content.SharedPreferences
import java.util.Calendar

fun Context.prefs(): SharedPreferences = getSharedPreferences("morningcall", Context.MODE_PRIVATE)

/** 안드로이드 알람 등록. 화면이 꺼져 있거나 앱이 닫혀 있어도 정해진 시각에 AlarmReceiver가 실행된다. */
object AlarmScheduler {
    private const val KEY_TRIGGER_AT = "alarm_trigger_at"
    private const val REQ_ALARM = 100
    private const val REQ_SHOW = 101

    /** 오늘 hour:minute이 이미 지났으면 내일 같은 시각 */
    fun nextTrigger(hour: Int, minute: Int): Long {
        val cal = Calendar.getInstance().apply {
            set(Calendar.HOUR_OF_DAY, hour)
            set(Calendar.MINUTE, minute)
            set(Calendar.SECOND, 0)
            set(Calendar.MILLISECOND, 0)
        }
        if (cal.timeInMillis <= System.currentTimeMillis()) cal.add(Calendar.DAY_OF_YEAR, 1)
        return cal.timeInMillis
    }

    /** @return true면 정확한 시각 알람, false면 시스템이 막아서 부정확한 알람으로 등록됨 */
    fun schedule(context: Context, triggerAt: Long): Boolean {
        val alarmManager = context.getSystemService(AlarmManager::class.java)
        context.prefs().edit().putLong(KEY_TRIGGER_AT, triggerAt).apply()
        val operation = alarmIntent(context)
        return try {
            // setAlarmClock은 절전 모드에서도 정확히 울리고, 상태 표시줄에 알람 아이콘도 보여 준다
            val show = PendingIntent.getActivity(
                context, REQ_SHOW, Intent(context, MainActivity::class.java), PendingIntent.FLAG_IMMUTABLE,
            )
            alarmManager.setAlarmClock(AlarmManager.AlarmClockInfo(triggerAt, show), operation)
            true
        } catch (e: SecurityException) {
            alarmManager.setAndAllowWhileIdle(AlarmManager.RTC_WAKEUP, triggerAt, operation)
            false
        }
    }

    fun cancel(context: Context) {
        context.getSystemService(AlarmManager::class.java).cancel(alarmIntent(context))
        clearSaved(context)
    }

    fun clearSaved(context: Context) {
        context.prefs().edit().remove(KEY_TRIGGER_AT).apply()
    }

    /** 재부팅하면 등록된 알람이 사라지므로 저장해 둔 시각으로 다시 등록 */
    fun rescheduleSaved(context: Context) {
        val triggerAt = context.prefs().getLong(KEY_TRIGGER_AT, 0L)
        if (triggerAt > System.currentTimeMillis()) schedule(context, triggerAt)
    }

    private fun alarmIntent(context: Context): PendingIntent = PendingIntent.getBroadcast(
        context, REQ_ALARM, Intent(context, AlarmReceiver::class.java),
        PendingIntent.FLAG_UPDATE_CURRENT or PendingIntent.FLAG_IMMUTABLE,
    )
}
