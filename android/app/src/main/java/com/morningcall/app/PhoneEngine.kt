package com.morningcall.app

import android.content.Context
import com.chaquo.python.PyObject
import com.chaquo.python.Python
import com.chaquo.python.android.AndroidPlatform
import org.json.JSONObject
import java.util.concurrent.Executors

/**
 * 서버(main.py)와 같은 대화 엔진(engine/dialog.py, Python)을 폰에서 돌린다 (Chaquopy).
 * AI는 PhoneAi(Gemma)를 쓰고, 결과는 서버와 같은 형식의 이벤트로 화면에 넘긴다.
 */
object PhoneEngine {
    // 대화 요청은 한 번에 하나씩 차례로 처리한다 (폰 AI도 한 번에 한 대화만 생성한다)
    private val worker = Executors.newSingleThreadExecutor()
    private val cancelled = mutableSetOf<String>()

    private fun module(ctx: Context): PyObject {
        if (!Python.isStarted()) Python.start(AndroidPlatform(ctx.applicationContext))
        return Python.getInstance().getModule("phone")
    }

    /** 앱을 열 때 Python을 미리 띄워 둔다 (처음 띄우는 데 1~2초 걸린다) */
    fun warmUp(ctx: Context) = worker.execute { runCatching { module(ctx) } }

    /** Python이 이벤트마다 부른다 */
    class Emitter(private val send: (String) -> Unit) {
        fun emit(json: String) = send(json)
    }

    /**
     * kind: "initial"(첫 인사) | "chat"(대화 한 턴). 결과는 send(id, 이벤트 JSON)로 보낸다.
     * 첫 인사는 {"type":"initial", message, user_content}, 대화는 meta → sentence… → done, 실패하면 error
     */
    fun request(ctx: Context, id: String, kind: String, payload: String, send: (String, String) -> Unit) {
        worker.execute {
            val emit = { json: String -> if (synchronized(cancelled) { id !in cancelled }) send(id, json) }
            try {
                if (kind == "initial") {
                    val result = JSONObject(module(ctx).callAttr("initial", payload, PhoneAi).toString())
                    emit(result.put("type", "initial").toString())
                } else {
                    module(ctx).callAttr("chat", payload, PhoneAi, Emitter(emit))
                }
            } catch (e: Throwable) {
                emit(JSONObject().put("type", "error").put("detail", "폰 AI 오류: ${e.message}").toString())
            } finally {
                synchronized(cancelled) { cancelled -= id }
            }
        }
    }

    /** 사용자가 끼어들어 앞 대답을 멈출 때. 남은 이벤트는 버리고 생성도 멈춘다 */
    fun cancel(id: String) {
        synchronized(cancelled) { cancelled += id }
        PhoneAi.cancelCurrent()
    }
}
