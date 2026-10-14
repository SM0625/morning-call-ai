import java.net.Inet4Address
import java.net.NetworkInterface

plugins {
    id("com.android.application")
    id("org.jetbrains.kotlin.android")
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
    kotlinOptions {
        jvmTarget = "17"
    }
}
