# AI 모닝콜

정해 둔 시간이 되면 AI가 "전화를 걸어" 음성으로 대화하며 깨워 주는 웹 서비스입니다.
AI는 [Ollama](https://ollama.com)로 내 컴퓨터에서 직접 돌리기 때문에 API 키나 비용이 들지 않습니다.

## 주요 기능

- **알람 설정**: 이름, 오늘 일정, 알람 시간, 사용할 AI 모델을 입력합니다.
- **AI 음성 통화**: 알람 시간이 되면 AI가 먼저 인사를 건넵니다. 브라우저 음성 합성(TTS)으로 말하고, 음성 인식(STT)으로 대답을 듣습니다. 마이크를 쓸 수 없으면 텍스트로도 대답할 수 있습니다.
- **4단계 대화**: 기상 확인 → 오늘 일정 안내 → 날씨 안내와 침대에서 나오기 → 마무리 순으로 진행됩니다.
  사용자의 대답을 "버팀 / 깸 / 침대에서 나옴 / 딴 얘기"로 분류해서, 실제로 일어났다고 할 때만 다음 단계로 넘어갑니다.
  버티거나 핑계를 대면 단계를 유지한 채 계속 깨웁니다.
- **누군가 깨워 주는 느낌**: 매 답변에 "전화 안 끊을 거예요", "목소리가 아직 잠겨 있네요"처럼 전화로 깨우는 중이라는 표현을 바꿔 가며 넣습니다.
- **답변 품질 검사**: 반말, 영어·한자 섞임, "더 주무세요" 같은 포기 발언, 같은 말 반복이 있으면 문제를 알려 주고 한 번 다시 생성합니다.
- **빠른 첫 응답**: 알람 2분 전에 AI 모델을 미리 메모리에 올려 둡니다.
- **끄기 방지**: 사용자가 일어났다고 AI가 판단해야 통화 종료 버튼이 활성화됩니다. 다시 알림(스누즈)을 누르면 5분 뒤에 또 울립니다.
- **날씨**: 브라우저 위치 정보로 [Open-Meteo](https://open-meteo.com)에서 현재 날씨를 가져옵니다. 위치 권한을 허용하지 않으면 서울 날씨를 씁니다.

## 구성

| 파일 | 설명 |
|---|---|
| `main.py` | FastAPI 백엔드. Ollama를 호출해 AI 답변을 만듭니다. |
| `static/index.html` | 프론트엔드 (알람 설정, 통화 화면, 음성 처리, 날씨) |
| `requirements.txt` | Python 패키지 목록 |
| `run.bat` | Windows용 실행 스크립트 (패키지 설치 + 서버 실행) |

### API

| 메서드 | 경로 | 설명 |
|---|---|---|
| `GET` | `/` | 웹 화면 |
| `GET` | `/api/initial-message` | 알람이 울릴 때 AI가 건네는 첫 인사 |
| `POST` | `/api/chat` | 대화 한 턴. 답변, 종료 가능 여부, 대화 단계를 돌려줍니다. |
| `GET` | `/api/models` | 설치된 Ollama 모델 목록 |
| `POST` | `/api/warmup` | 알람 직전에 모델을 메모리에 미리 올림 |

## 설치하는 법

### 1. Python 설치

Python 3.10 이상이 필요합니다. <https://www.python.org/downloads/>에서 받을 수 있고, 설치할 때 **"Add Python to PATH"**를 체크하세요.

### 2. Ollama 설치

<https://ollama.com>에서 받아 설치합니다. winget을 쓴다면 아래 명령으로도 설치할 수 있습니다.

```powershell
winget install --id Ollama.Ollama -e
```

### 3. AI 모델 받기

기본 모델은 `gemma3:4b`(약 3.3GB)입니다. 더 가볍게 쓰려면 한국어 특화 모델 `exaone3.5:2.4b`(약 1.6GB)도 화면에서 고를 수 있습니다.

```powershell
ollama pull gemma3:4b
```

### 4. Python 패키지 설치

프로젝트 폴더에서 실행합니다.

```powershell
pip install -r requirements.txt
```

## 실행하는 법

1. **Ollama 실행**: Ollama 앱을 켜 두거나, 터미널을 하나 따로 열어 아래 명령을 실행합니다.

   ```powershell
   ollama serve
   ```

   이미 실행 중이라는 오류가 나오면 그대로 두면 됩니다.

2. **서버 실행**: 프로젝트 폴더의 `run.bat`을 더블클릭합니다. 터미널에서 직접 실행하려면 아래 명령을 씁니다.

   ```powershell
   python -m uvicorn main:app --host 0.0.0.0 --port 8000 --reload
   ```

3. **브라우저 접속**: <http://localhost:8000>을 엽니다.
   - 음성 인식이 잘 되는 **Chrome**이나 **Edge**를 권장합니다.
   - 마이크와 위치 권한을 허용해 주세요.
   - 알람이 울리기 전까지 브라우저 탭을 열어 두어야 합니다.

4. **종료**: 서버 터미널에서 `Ctrl + C`를 누릅니다.

## 설정

- **다른 모델 사용**: 웹 화면에서 모델을 고르거나, 환경 변수 `OLLAMA_MODEL`로 기본 모델을 바꿀 수 있습니다.

  ```powershell
  ollama pull exaone3.5:2.4b
  $env:OLLAMA_MODEL = "exaone3.5:2.4b"; python -m uvicorn main:app --port 8000
  ```

- **GPU 레이어 수**: 기본으로 모델의 모든 레이어를 GPU에 올립니다(`OLLAMA_NUM_GPU=99`). VRAM이 부족하면 자동으로 기본 설정으로 다시 시도합니다.
- `.env.example`에 있는 `ANTHROPIC_API_KEY`는 현재 코드에서 쓰지 않습니다.

## 문제 해결

| 증상 | 해결 |
|---|---|
| "Ollama가 실행되지 않고 있습니다" | `ollama serve`를 실행하거나 Ollama 앱을 켭니다. |
| "model not found" 오류 | `ollama pull gemma3:4b`로 모델을 받습니다. |
| 음성 인식이 안 됨 | Chrome이나 Edge를 쓰고 마이크 권한을 허용합니다. 안 되면 텍스트로 입력하세요. |
| `ollama` 명령을 찾을 수 없음 | Ollama를 설치한 뒤 터미널을 새로 엽니다. |
| 포트 8000을 이미 사용 중 | 명령의 `--port 8000`을 `--port 8001` 등으로 바꿉니다. |
