package com.morningcall.app

import android.app.AlarmManager
import android.app.PendingIntent
import android.content.Context
import android.content.Intent
import android.content.SharedPreferences
import java.util.Calendar

fun Context.prefs(): SharedPreferences = getSharedPreferences("morningcall", Context.MODE_PRIVATE)

/**
 * 안드로이드 알람 등록. 화면이 꺼져 있거나 앱이 닫혀 있어도 정해진 시각에 AlarmReceiver가 실행된다.
 *
 * 반복 알람: 요일 묶음(daysMask)의 비트 i가 켜져 있으면 그 요일에 울린다 (i: 0 일요일 … 6 토요일, JS의 Date.getDay()와 같다).
 * 0이면 한 번만 울린다. 안드로이드에는 '매주 이 요일' 알람이 따로 없어서, 울릴 때마다 AlarmReceiver가 다음 번을 등록한다.
 */
object AlarmScheduler {
    private const val KEY_TRIGGER_AT = "alarm_trigger_at"
    private const val KEY_HOUR = "alarm_hour"
    private const val KEY_MINUTE = "alarm_minute"
    private const val KEY_DAYS = "alarm_days"
    private const val REQ_ALARM = 100
    private const val REQ_SHOW = 101

    /** daysMask에 든 요일 중 hour:minute이 아직 오지 않은 가장 가까운 때. 0이면 오늘(지났으면 내일) */
    fun nextTrigger(hour: Int, minute: Int, daysMask: Int = 0): Long {
        val now = System.currentTimeMillis()
        for (add in 0..7) {
            val cal = Calendar.getInstance().apply {
                add(Calendar.DAY_OF_YEAR, add)
                set(Calendar.HOUR_OF_DAY, hour)
                set(Calendar.MINUTE, minute)
                set(Calendar.SECOND, 0)
                set(Calendar.MILLISECOND, 0)
            }
            if (cal.timeInMillis <= now) continue
            val dow = cal.get(Calendar.DAY_OF_WEEK) - 1  // 0 일요일 … 6 토요일
            if (daysMask == 0 || (daysMask shr dow) and 1 == 1) return cal.timeInMillis
        }
        return nextTrigger(hour, minute, 0)  // 요일 묶음이 잘못됐을 때(0x7F 밖) 한 번 알람으로
    }

    /** 사용자가 맞춘 알람(시각과 반복 요일)을 기억해 두고 다음 번을 등록한다 */
    fun setAlarm(context: Context, hour: Int, minute: Int, daysMask: Int): Pair<Long, Boolean> {
        context.prefs().edit().putInt(KEY_HOUR, hour).putInt(KEY_MINUTE, minute).putInt(KEY_DAYS, daysMask and 0x7F).apply()
        val at = nextTrigger(hour, minute, daysMask and 0x7F)
        return at to schedule(context, at)
    }

    /** 반복 알람이면 다음 번 시각, 아니면 null */
    fun nextRepeat(context: Context): Long? {
        val p = context.prefs()
        val days = p.getInt(KEY_DAYS, 0)
        if (days == 0 || !p.contains(KEY_HOUR)) return null
        return nextTrigger(p.getInt(KEY_HOUR, 7), p.getInt(KEY_MINUTE, 0), days)
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

    /** 알람 끄기: 반복도 함께 지운다 */
    fun cancel(context: Context) {
        context.getSystemService(AlarmManager::class.java).cancel(alarmIntent(context))
        context.prefs().edit().remove(KEY_TRIGGER_AT).remove(KEY_HOUR).remove(KEY_MINUTE).remove(KEY_DAYS).apply()
    }

    /**
     * 알람이 울렸을 때: 반복 알람이면 다음 번을 바로 등록하고, 한 번 알람이면 지운다.
     * 스누즈(snooze)는 같은 자리에 '5분 뒤'를 덮어쓰지만, 스누즈가 울릴 때 여기서 다시 다음 번을 등록하므로 반복은 이어진다.
     */
    fun onFired(context: Context) {
        val next = nextRepeat(context)
        if (next != null) schedule(context, next)
        else context.prefs().edit().remove(KEY_TRIGGER_AT).apply()
    }

    /** 재부팅하면 등록된 알람이 사라지므로 다시 등록. 꺼져 있는 동안 지난 반복 알람은 다음 번으로 */
    fun rescheduleSaved(context: Context) {
        val triggerAt = context.prefs().getLong(KEY_TRIGGER_AT, 0L)
        when {
            triggerAt > System.currentTimeMillis() -> schedule(context, triggerAt)
            else -> nextRepeat(context)?.let { schedule(context, it) }
        }
    }

    private fun alarmIntent(context: Context): PendingIntent = PendingIntent.getBroadcast(
        context, REQ_ALARM, Intent(context, AlarmReceiver::class.java),
        PendingIntent.FLAG_UPDATE_CURRENT or PendingIntent.FLAG_IMMUTABLE,
    )
}
