# Troubleshooting Guide

이 문서는 기존 `team_guide.md`, `data_collection_guide.md`, `PIPELINE.md`, 수집 스크립트에 흩어져 있던 문제 해결 기록을 한곳에 정리한 것입니다.

## 빠른 점검 순서

1. DOBOT 전원 어댑터와 USB가 모두 연결되어 있는지 확인한다.
2. DOBOT 표시등이 초록불인지 확인한다. 빨간불이면 수집 프로그램에서 `X`를 눌러 알람을 해제한다.
3. 카메라 2대가 모두 인식되는지 확인한다.
4. 데이터 수집/추론 모두 현재 기준으로 `--cam1=wrist`, `--cam2=top` 매핑을 맞춘다.
5. 추론 문제는 먼저 서버 health check부터 확인한다.

```bash
curl http://[서버IP]:8000/health
```

## DOBOT 연결

| 증상 | 원인 후보 | 해결 |
|---|---|---|
| `DOBOT not found` | USB 인식 실패, 전원 미연결 | USB를 다시 꽂고 전원 어댑터를 확인한다. |
| DOBOT 연결 실패 | USB만 연결하고 전원 어댑터가 빠짐 | DOBOT은 전원 어댑터가 필요하다. |
| 잘못된 포트 감지 | 여러 USB serial 장치가 연결됨 | `--port /dev/cu.usbserial-XXXX` 또는 `--port /dev/tty.usbserial-XXXX`로 직접 지정한다. |
| DOBOT 빨간불 | 알람/리밋/비정상 상태 | 데이터 수집 프로그램에서 `X`를 눌러 알람 해제. 그래도 안 되면 USB 재연결 후 다시 실행한다. |
| DOBOT이 이상하게 움직임 | 기준 위치가 틀어짐, unsafe pose | `Q`로 홈 위치 `(200, 0, 50, 0)` 이동. 계속 이상하면 `A`로 리밋스위치 홈잉. |
| 실행 중 DOBOT 연결 끊김 | USB/시리얼 세션 불안정 | 수집 프로그램에서 `F`로 DOBOT 재연결. 실패하면 USB를 뽑았다가 다시 꽂는다. |

## 카메라

| 증상 | 해결 |
|---|---|
| 카메라 프리뷰가 안 뜸 | USB를 다시 꽂고 `--cam1`, `--cam2` 번호를 바꿔 실행한다. |
| 카메라 번호가 헷갈림 | 아래 확인 스크립트로 각 번호를 확인한다. |
| 키 입력이 안 됨 | OpenCV 프리뷰 창을 클릭해서 포커스를 준 뒤 키를 누른다. |
| 프리뷰가 조금 느림 | 정상일 수 있다. 수집 스크립트는 시리얼 병목을 줄이려고 포즈를 5프레임마다 갱신한다. |

현재 코드 기준 카메라 매핑:

| 옵션 | 의미 |
|---|---|
| `--cam1` | wrist 카메라 |
| `--cam2` | top 카메라 |

카메라 번호 확인:

```bash
python -c "
import cv2
for i in range(5):
    cap = cv2.VideoCapture(i)
    if cap.isOpened():
        ret, frame = cap.read()
        if ret:
            print(f'카메라 {i}: OK ({frame.shape[1]}x{frame.shape[0]})')
            cv2.imshow(f'Camera {i}', frame)
        cap.release()
cv2.waitKey(3000)
cv2.destroyAllWindows()
"
```

## 데이터 수집

| 증상 | 해결 |
|---|---|
| 기존 데이터가 사라짐 | `--resume` 없이 같은 `--save_dir`로 실행하면 기존 데이터가 삭제된다. 이어서 수집할 때는 반드시 `--resume`을 붙인다. |
| 수집 중 프로그램이 꺼짐 | 같은 `--save_dir`에 `--resume`으로 다시 실행한 뒤 복구 안내가 뜨면 `1`을 눌러 복구한다. |
| 미저장 에피소드가 남음 | 저장하려면 `V`, 폐기하려면 `D`. |
| 직전 스텝만 실수함 | `W`로 마지막 스텝을 되돌린다. |
| 실패한 에피소드를 저장함 | 가능하면 저장 전에 `D`로 버린다. 잘못된 데이터는 학습 품질을 크게 떨어뜨린다. |
| delta 값이 튐 | 팔을 천천히 움직이고 한 스텝을 너무 크게 잡지 않는다. |

검증과 자동 수정:

```bash
python scripts/03_validate_dataset.py --dataset_dir ./데이터셋 --fix
```

## 데이터셋 검증 에러

| 증상 | 해결 |
|---|---|
| `videos/ only -- this is v2.x format` | `python scripts/02_convert_v2_to_v3.py`로 v3 형식으로 변환한다. |
| `tasks.parquet not found` | `03_validate_dataset.py --fix`를 실행한다. |
| `meta/info.json` 누락 | `03_validate_dataset.py --fix`로 복구 가능한지 먼저 확인한다. |
| `FileNotFoundError: ... .jpg` | 서버 전송 시 이미지 폴더가 빠졌을 가능성이 높다. `scp -r`로 데이터셋 전체를 다시 전송한다. |
| image path not found | parquet의 이미지 경로와 실제 파일 위치가 다르다. validate `--fix` 후에도 남으면 데이터셋을 다시 전송/수집한다. |

## 학습

| 증상 | 해결 |
|---|---|
| `lerobot-train: command not found` | 서버에서 `source .venv/bin/activate` 또는 올바른 conda/venv 활성화를 먼저 한다. |
| `CUDA out of memory` | `train.sh`의 batch size를 줄인다. 예: `4 -> 2 -> 1`. |
| 학습이 멈춘 듯 보임 | 다른 터미널에서 `nvidia-smi`로 GPU 사용량을 확인한다. |
| 이어서 학습이 안 됨 | 같은 출력 경로에 `resume` 인자를 붙여 실행해야 한다. 출력 경로가 다르면 체크포인트를 찾지 못한다. |
| loss가 줄지 않음 | 실패 에피소드 제거, 데이터 추가 수집, 물체 위치 다양화, 카메라 고정 상태를 확인한다. |

학습 resume 예시:

```bash
./train.sh ./tissue_dataset_v1 1 20000 outputs/tissue_v1 resume
```

## 추론 서버

| 증상 | 해결 |
|---|---|
| 서버 연결 안 됨 | 서버 IP/포트 확인 후 `curl http://서버IP:8000/health`를 실행한다. |
| `model not found` 또는 config 없음 | `PI0_MODEL_PATH`가 `.../checkpoints/last/pretrained_model` 또는 실제 checkpoint의 `pretrained_model`을 가리키는지 확인한다. |
| LoRA 로딩 실패 | base 모델 접근 권한, `peft` 설치, checkpoint 경로를 확인한다. |
| 서버는 켜졌는데 응답이 느림 | 첫 요청은 warm-up 때문에 느릴 수 있다. 이후에도 느리면 GPU 사용량과 네트워크를 확인한다. |
| HTTP timeout | 서버 추론이 10초를 넘었거나 네트워크가 끊긴 상태다. health check 후 서버 로그를 확인한다. |

서버 실행 예시:

```bash
PI0_POLICY_TYPE=pi0_fast \
PI0_MODEL_PATH=./outputs/tissue_v1/checkpoints/last/pretrained_model \
python server/pi0_server.py
```

## 추론 클라이언트 / 로봇 실행

| 증상 | 해결 |
|---|---|
| 로봇이 너무 크게 움직임 | `--chunk-size 1`로 실행하고, 자동 모드 전에 `R` 또는 1회 추론으로 확인한다. |
| 동작이 불안정함 | 자동 모드를 끄고 한 스텝씩 확인한다. 데이터 추가 수집과 재학습이 필요할 수 있다. |
| 그리퍼가 늦거나 안 잡힘 | gripper delay를 0.2~0.5초로 늘려 현장 테스트한다. |
| WebSocket 클라이언트에서 `websocket` 모듈 없음 | `pip install websocket-client`를 실행한다. |
| 명령어가 모델 학습 문장과 다름 | 학습 때 쓴 task prompt와 최대한 같은 문장을 사용한다. |

보수적인 추론 실행:

```bash
python client/pi0_dobot_client.py \
    --server http://[서버IP]:8000 \
    --cam1 0 --cam2 1 \
    --task "pick up the tissue" \
    --chunk-size 1
```

## 그리퍼 현장 디버깅 기록

추론 중 그리퍼가 기대한 타이밍에 작동하지 않으면 아래 값을 조정한다.

| 값 | 기본 의미 | 조정 방향 |
|---|---|---|
| gripper delay after move | 이동 후 그리퍼 명령까지 대기 | 0.1초에서 0.3초 정도로 증가 |
| gripper action wait | 그리퍼 동작 완료 대기 | 0.5초 전후로 유지 또는 증가 |
| grip threshold | `action[4] > threshold`면 grip ON | 모델 출력 분포를 보고 0.5 기준을 조정 |

현재 모듈화 코드에서는 `dobot_vla.infrastructure.dobot.DobotConfig`의 `gripper_delay_s`, `gripper_wait_s`에서 조정한다.

## 음성 / LLM

| 증상 | 해결 |
|---|---|
| STT 결과가 너무 짧거나 이상함 | 조용한 환경에서 다시 말하고, 마이크 입력 장치를 확인한다. |
| 명령이 대화로 분류됨 | 사용 가능한 물체명(과자, 음료, 연필, 지우개, 휴지, 스트레스볼)을 포함해 말한다. |
| 물체명이 인식되지 않음 | `dobot_vla/domain/tasks.py`의 `COMMAND_MAP`에 alias를 추가한다. |
| LLM 모델 로딩 실패 | 장비 메모리와 모델명을 확인한다. 빠른 테스트는 rule/simple backend로 먼저 진행한다. |

## 안전 원칙

- 자동 모드 전에 항상 한 스텝 추론으로 방향을 확인한다.
- 카메라나 물체 배치를 바꾸면 기존 학습 데이터와 분포가 달라질 수 있다.
- 로봇 좌표는 최종 target 기준으로 safety bounds에 clamp된다.
- 이상 동작이 보이면 즉시 자동 모드를 끄고 홈 위치로 보낸다.
