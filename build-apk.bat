@echo off
chcp 65001 >nul
setlocal
rem AI 모닝콜 APK 빌드. 이 PC의 주소를 앱 기본 서버 주소로 넣고, 결과를 바탕화면에 복사한다.
rem 다른 주소를 넣으려면: build-apk.bat http://주소:8000

set "PROJECT=%~dp0"
if "%PROJECT:~-1%"=="\" set "PROJECT=%PROJECT:~0,-1%"

rem 빌드 도구(Gradle 8.7)는 JDK 17에서만 돈다. 시스템 JAVA_HOME이 다른 버전이어도 JDK 17이 있으면 그걸 쓴다
for /d %%D in ("C:\Program Files\Microsoft\jdk-17*" "C:\Program Files\Eclipse Adoptium\jdk-17*") do set "JAVA_HOME=%%~D"
if not defined ANDROID_HOME set "ANDROID_HOME=%LOCALAPPDATA%\Android\Sdk"
set "GRADLE="
for /d %%D in ("%LOCALAPPDATA%\Android\gradle\gradle-*") do set "GRADLE=%%D\bin\gradle.bat"

if not defined JAVA_HOME (
  echo [오류] JDK 17을 찾지 못했어요. 설치: winget install EclipseAdoptium.Temurin.17.JDK
  goto :fail
)
if not exist "%ANDROID_HOME%\platforms" (
  echo [오류] Android SDK를 찾지 못했어요: %ANDROID_HOME%
  goto :fail
)
if not defined GRADLE (
  echo [오류] Gradle을 찾지 못했어요: %LOCALAPPDATA%\Android\gradle
  goto :fail
)

if not exist "%PROJECT%\android\local.properties" (
  >"%PROJECT%\android\local.properties" echo sdk.dir=%ANDROID_HOME:\=\\%
)

set "SERVER_ARG="
if not "%~1"=="" set "SERVER_ARG=-PserverUrl=%~1"

rem 경로에 한글이 있으면 안드로이드 빌드 도구가 실패할 수 있어 임시 영문 드라이브로 연결해서 빌드
set "DRIVE=M:"
subst %DRIVE% /D >nul 2>nul
subst %DRIVE% "%PROJECT%"
pushd %DRIVE%\android
call "%GRADLE%" assembleDebug --console=plain %SERVER_ARG%
set "RESULT=%ERRORLEVEL%"
popd
subst %DRIVE% /D

if not "%RESULT%"=="0" (
  echo [오류] 빌드에 실패했어요. 위의 메시지를 확인해 주세요.
  goto :fail
)

rem 빌드 결과는 OneDrive 밖에 있다 (android\build.gradle.kts). 바탕화면은 OneDrive 아래일 수 있어 실제 위치를 묻는다
for /f "usebackq delims=" %%P in (`powershell -NoProfile -Command "[Environment]::GetFolderPath('Desktop')"`) do set "DESKTOP=%%P"
copy /Y "%LOCALAPPDATA%\morningcall-build\app\outputs\apk\debug\app-debug.apk" "%DESKTOP%\AI모닝콜.apk" >nul || (
  echo [오류] APK를 바탕화면에 복사하지 못했어요: %DESKTOP%
  goto :fail
)
echo.
echo  완료: 바탕화면의 AI모닝콜.apk 를 폰에 설치하세요.
echo.
pause
exit /b 0

:fail
pause
exit /b 1
