import java.net.Inet4Address
import java.net.NetworkInterface

import org.jetbrains.kotlin.gradle.dsl.JvmTarget

plugins {
    id("com.android.application")
    id("org.jetbrains.kotlin.android")
    id("com.chaquo.python")
}

// 앱에 기본으로 넣을 PC 서버 주소. -PserverUrl=http://... 로 직접 지정하거나,
// 지정하지 않으면 빌드하는 PC의 IPv4 주소를 찾아 http://<IP>:8000 으로 넣는다.
fun defaultServerUrl(): String {
    (findProperty("serverUrl") as String?)?.takeIf { it.isNotBlank() }?.let { return it }
    val virtual = Regex("virtual|vethernet|vmware|hyper-v|loopback|bluetooth", RegexOption.IGNORE_CASE)
    val ip = NetworkInterface.getNetworkInterfaces().toList()
        .filter { it.isUp && !it.isLoopback && !it.isVirtual && !virtual.containsMatchIn(it.displayName) }
        .flatMap { it.inetAddresses.toList() }
        .filterIsInstance<Inet4Address>()
        .firstOrNull { !it.isLinkLocalAddress }
        ?.hostAddress
    return if (ip != null) "http://$ip:8000" else ""
}

val serverUrl = defaultServerUrl()
println("앱 기본 서버 주소: ${serverUrl.ifEmpty { "(찾지 못함 — 앱에서 직접 입력)" }}")

android {
    namespace = "com.morningcall.app"
    compileSdk = 34

    defaultConfig {
        applicationId = "com.morningcall.app"
        minSdk = 26
        targetSdk = 34
        versionCode = 1
        versionName = "1.0"
        buildConfigField("String", "DEFAULT_SERVER_URL", "\"$serverUrl\"")
        // 폰 AI(LiteRT-LM)는 64비트 ARM 폰용 라이브러리만 있다. 에뮬레이터용 x86_64도 넣는다
        ndk { abiFilters += listOf("arm64-v8a", "x86_64") }
    }

    buildFeatures {
        buildConfig = true
    }

    // 웹과 같은 화면(static/index.html)을 그대로 앱에 넣는다
    sourceSets["main"].assets.srcDirs("../../static")

    compileOptions {
        sourceCompatibility = JavaVersion.VERSION_17
        targetCompatibility = JavaVersion.VERSION_17
    }
}

kotlin {
    compilerOptions { jvmTarget.set(JvmTarget.JVM_17) }
}

chaquopy {
    defaultConfig { version = "3.12" }
    // 서버(main.py)와 같은 대화 엔진을 그대로 넣는다
    sourceSets { getByName("main") { srcDir("../../engine") } }
}

dependencies {
    // 폰에서 Gemma 모델을 돌리는 Google 공식 라이브러리
    implementation("com.google.ai.edge.litertlm:litertlm-android:0.19.1")
}
