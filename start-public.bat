@echo off
chcp 65001 >nul
setlocal
rem AI 모닝콜 서버를 켜고 ngrok 고정 주소로 인터넷에 공개한다. 앱(APK)은 이 주소로 접속한다.
rem 처음 설치하는 법은 서버-실행법.md 를 보세요.

set "DOMAIN=cheese-wiring-knoll.ngrok-free.dev"
set "MODEL=gemma3:4b"
cd /d "%~dp0"

echo.
echo  ========================================
echo   AI 모닝콜 공개 서버
echo   주소: https://%DOMAIN%
echo  ========================================
echo.

where python >nul 2>nul || (echo [오류] Python을 찾지 못했어요. 서버-실행법.md 의 1단계를 확인하세요. & goto :fail)
where ollama >nul 2>nul || (echo [오류] Ollama를 찾지 못했어요. 설치했다면 이 창을 닫고 다시 실행하세요. & goto :fail)
where ngrok >nul 2>nul || (echo [오류] ngrok을 찾지 못했어요. 설치했다면 이 창을 닫고 다시 실행하세요. & goto :fail)

echo  [1/5] Python 패키지 확인
python -m pip install -r requirements.txt --quiet --disable-pip-version-check || goto :fail

echo  [2/5] Ollama 확인
ollama list >nul 2>nul
if errorlevel 1 (
  start "Ollama" /min ollama serve
  timeout /t 5 /nobreak >nul
)
ollama list | findstr /c:"%MODEL%" >nul || (
  echo        %MODEL% 모델이 없어서 받는 중이에요. 약 3.3GB라 몇 분 걸려요.
  ollama pull %MODEL% || goto :fail
)

echo  [3/5] ngrok 확인
rem winget으로 받은 ngrok은 버전이 오래돼서 그대로 쓰면 접속이 거부된다
ngrok update >nul 2>nul
findstr /c:"authtoken" "%LOCALAPPDATA%\ngrok\ngrok.yml" >nul 2>nul || (
  echo        ngrok 토큰이 등록되어 있지 않아요. 주소 주인의 ngrok 토큰을 붙여 넣으세요.
  set /p "TOKEN=       토큰: "
  call ngrok config add-authtoken %%TOKEN%% || goto :fail
)

echo  [4/5] 서버 실행 (새 창)
start "AI 모닝콜 서버" python -m uvicorn main:app --host 0.0.0.0 --port 8000

echo  [5/5] 인터넷 공개
echo.
echo  이 창과 "AI 모닝콜 서버" 창을 닫지 마세요. 끄려면 두 창을 모두 닫으면 됩니다.
echo  기록 보기: http://127.0.0.1:8000/records  (이 PC에서만)
echo  "already online" 오류가 나오면 다른 PC에서 이미 서버를 켜 둔 거예요. 그쪽을 먼저 끄세요.
echo.
ngrok http 8000 --url=%DOMAIN%

:fail
echo.
pause
exit /b 1
