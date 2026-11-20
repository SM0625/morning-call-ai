// 프로젝트가 OneDrive 안에 있으면 동기화가 빌드 파일을 잡고 있어 "삭제/접근 거부"로 빌드가 실패한다.
// Windows에서는 빌드 결과를 OneDrive 밖(%LOCALAPPDATA%\morningcall-build)에 둔다
System.getenv("LOCALAPPDATA")?.let { local ->
    allprojects { layout.buildDirectory.set(File(local, "morningcall-build/$name")) }
}

plugins {
    id("com.android.application") version "8.5.2" apply false
    // 폰 AI(LiteRT-LM)가 Kotlin 2.4로 빌드되어 있어 2.3 이상이 필요하다
    id("org.jetbrains.kotlin.android") version "2.4.21" apply false
    // 대화 엔진(engine/, Python)을 서버와 폰이 같이 쓰도록 폰에서 Python을 돌린다
    id("com.chaquo.python") version "17.2.0" apply false
}
