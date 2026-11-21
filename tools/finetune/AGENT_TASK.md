# 작업 지시서: Gemma 4 E2B 깨우기 대화 미세조정 (Google Colab)

이 문서는 브라우저를 다룰 수 있는 AI 에이전트가 사람 대신 Colab에서 학습을 돌리도록 쓴 것이다.
결과는 사람(학생)에게 한국어로, 쉬운 말로 보고한다.

## 목표

모닝콜 앱의 폰 AI 모델 `Gemma 4 E2B`를 '깨우기 대화' 데이터로 LoRA 미세조정하고, 폰에서 돌릴 수 있는 `.litertlm` 파일을 만든다.

## 받은 파일 (`morning-gemma-finetune.zip`)

| 경로 | 내용 |
|---|---|
| `tools/finetune/gemma4_wake_lora.ipynb` | 실행할 노트북. 위에서부터 차례로 실행하면 끝까지 간다 |
| `tools/finetune/wake_train.jsonl` | 학습 데이터: 가상 통화 274개, AI 대답 1,521개. **고치지 말 것** |
| `engine/dialog.py`, `engine/weather_quiz.py` | 대화 엔진. 학습 전후 대답을 앱과 같은 프롬프트로 비교·검사하는 데 쓴다 |
| `AGENT_TASK.md` | 이 문서 |

## 순서

1. https://colab.research.google.com 에서 **파일 → 노트북 업로드**로 `gemma4_wake_lora.ipynb`를 연다.
2. **런타임 → 런타임 유형 변경 → GPU**(T4 이상). 고용량 RAM을 고를 수 있으면 고른다(8번 변환 단계에서 메모리를 많이 쓴다).
3. 왼쪽 **파일(📁)** 창에 `morning-gemma-finetune.zip`을 그대로(압축을 풀지 말고) 올린다. 2번 칸이 알아서 푼다.
4. 0번 칸부터 마지막 칸까지 **차례로 하나씩** 실행한다. 각 칸이 끝난 것을 확인하고 다음으로 간다.
5. 9번 칸에서 구글 드라이브 연결을 허락하면 `내 드라이브/morning-call/morning-gemma.litertlm`이 생긴다.

## 칸별로 확인할 것

| 칸 | 정상 | 문제가 생기면 |
|---|---|---|
| 0 | `Tesla T4` 등 GPU 이름 | 런타임 유형을 GPU로 바꾸고 다시 |
| 1 | 오류 없이 설치, `litert-torch --help` 출력 | `uv tool install`이 실패하면 `!pip install -q litert-torch-nightly litert-lm`로 대신 설치 |
| 2 | `통화 274개 → 학습 약 247개, 확인용 27개` | zip을 올렸는지 확인 |
| 3 | 모델 불러오기 완료 | 메모리 부족이면 `load_in_4bit=True`인지 확인 |
| 4 | 학습 전 대답 6개 출력 | `enable_thinking` 인자 오류면 그 인자만 빼고 실행 |
| 5 | `대화 템플릿이 예상과 달라요`가 **나오지 않아야** 함 | 나오면 멈추고 오류 전문을 보고 |
| 6 | `loss`가 줄어듦 | 메모리 부족이면 `gradient_accumulation_steps=8`은 두고 `MAX_LEN`을 3072로 |
| 7 | 학습 전/후 대답과 검사 결과 | – |
| 8 | `merged/`와 `out/model.litertlm`(대략 2.5~4GB) 생김 | 아래 '변환이 안 될 때' |
| 9 | 드라이브에 파일 복사됨 | – |

### 변환(8번)이 안 될 때

- **메모리 부족으로 런타임이 다시 시작됨**: 고용량 RAM 런타임으로 바꿔 3번부터 다시 하거나, 합친 모델 `merged/` 폴더를 드라이브에 저장해 두고 보고한다.
- **`export_hf`의 옵션 이름이 다르다는 오류**: `!litert-torch export_hf --help`로 옵션을 확인해 맞춘다. 같은 일을 하는 다른 형식:
  ```
  !python -m litert_torch.generative.export_hf --hf_model_dir merged --externalize_embedder \
      --quantize dynamic_wi4_afp32 --cache_length 4096 --output out/model.litertlm
  ```
  이 형식은 대화 템플릿을 자동으로 바꾸지 않으므로, 먼저
  `https://huggingface.co/litert-community/gemma-4-E2B-it-litert-lm/resolve/main/chat_template.jinja`를 받아
  `merged/tokenizer_config.json`의 `chat_template` 값으로 넣고 `merged/chat_template.jinja`는 지운다.
  (원본 템플릿을 그대로 넣으면 폰에서 첫 메시지에 `map has no method named get` 오류가 난다.)
- `cache_length`는 4096보다 작게 하지 않는다. 앱의 시스템 프롬프트만 약 1,000토큰이다.

## 하지 말 것

- 학습 데이터(`wake_train.jsonl`)를 고치거나 다른 데이터를 섞지 않는다.
- 학습 설정(r=16, 2바퀴, learning_rate 1e-4)은 메모리 때문에 꼭 필요할 때만 바꾸고, 바꿨으면 보고한다.
- 결과가 나쁘게 나와도 숨기지 않는다. 확인하지 못한 것은 못 했다고 쓴다.

## 보고할 것 (사람에게, 한국어로)

1. 성공했는지, 드라이브의 파일 이름과 크기
2. 6번 칸의 `loss`, `eval_loss` 변화 (처음 → 끝)
3. 7번 칸 출력 전체 (학습 전/후 대답 6쌍과 검사 결과)
4. 8번 칸 마지막 `litert-lm run` 대답
5. 바꾼 설정이나 생긴 오류와 해결 방법
6. 걸린 시간

이 보고와 드라이브의 `morning-gemma.litertlm`을 받으면, 원래 작업자가 폰에 넣어 학습 전후를 측정한다
(`adb push morning-gemma.litertlm /sdcard/Android/data/com.morningcall.app/files/`).
