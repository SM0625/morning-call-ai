@echo off
chcp 65001 >nul
echo.
echo  ========================================
echo   AI 모닝콜 서버 (Ollama 무료 로컬 AI)
echo  ========================================
echo.
echo  [사전 준비] Ollama가 설치되어 있어야 합니다.
echo   - 설치: https://ollama.com 에서 다운로드
echo   - 모델 다운로드: ollama pull llama3.2
echo   - 실행: ollama serve  (별도 터미널)
echo.

pip install -r requirements.txt --quiet

echo  서버 주소: http://localhost:8000
echo  브라우저에서 위 주소를 열어주세요.
echo  종료: Ctrl+C
echo.

python -m uvicorn main:app --host 0.0.0.0 --port 8000 --reload
pause
