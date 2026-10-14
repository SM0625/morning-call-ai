package com.morningcall.app

import android.content.BroadcastReceiver
import android.content.Context
import android.content.Intent

/** 재부팅하면 등록된 알람이 지워지므로 다시 등록한다 */
class BootReceiver : BroadcastReceiver() {
    override fun onReceive(context: Context, intent: Intent) {
        if (intent.action == Intent.ACTION_BOOT_COMPLETED) AlarmScheduler.rescheduleSaved(context)
    }
}
