# DOBOT Magician × LeRobot × Pi0 Pipeline

**부산 로보틱스 AI 교육 프로그램 -- 단일 로봇 팔 임시학습(Imitation Learning) 파이프라인**

---

## 구조

```
+-----------------------------------------------------------------+
|                    TRAINING PIPELINE                            |
|                                                                 |
|  1) Collect Data ---> 2) Convert to v3 ---> 3) Validate ---> 4) Train |
|  (Single Arm,       (LeRobot v3.0       (Auto-fix      (Pi0    |
|   Sequential         Format)             Metadata)      -FAST) |
|   Teleoperation)                                                |
+-----------------------------------------------------------------+
|                    INFERENCE PIPELINE                           |
|                                                                 |
|  Pi0-FAST Server (A6000) --HTTP---> Client + DOBOT (local)      |
|                                                                 |
|  Model Output: Δ[x, y, z, r, grip]  (delta coordinates)       |
|  -> Inverse kinematics solved by DOBOT firmware (move_to API)   |
+-----------------------------------------------------------------+
```

## 설계

### 단일 암(리드암 부재시) 순차 방식

SO-100처럼 리더/팔로워가 없으므로 수동으로 팔을 위치시키며 프레임 단위로 수집.
`[S]` 관측 캡처 -> 팔 이동 -> `[E]` 델타 기록.

### 델타 액션

action은 절대 좌표가 아닌 변위값 `[Δx, Δy, Δz, Δr, grip]`으로 저장.

### 역기구학

Pi0 출력은 3D 직교좌표 delta. DOBOT 펌웨어가 `move_to(x, y, z, r)`로 역기구학 처리.

---

## 파일 구조

```
Dobot_VLM_VLA/
├── README.md
├── LICENSE
├── requirements.txt
├── train.sh                         # GPU 서버용 Pi0-FAST 학습 실행 래퍼
├── scripts/
│   ├── 01_collect_data.py           # 단계별 데이터 수집 (단일 팔 순차 방식)
│   ├── 02_convert_v2_to_v3.py       # v2.x → v3.0 포맷 변환
│   ├── 03_validate_dataset.py       # 데이터셋 검증 + 자동 수정
│   ├── 04_train_pi0.sh              # Pi0-FAST 학습 스크립트
│   ├── 05_inference_dobot.py        # 로컬 추론 테스트
│   ├── task_normalizer.py           # 태스크 정규화 유틸
│   └── test_dobot.py               # DOBOT 연결 테스트
├── server/
│   ├── pi0_server.py                # Pi0-FAST HTTP 추론 서버 (GPU)
│   └── pi0_ws_server.py             # WebSocket 스트리밍 서버
├── client/
│   ├── pi0_dobot_client.py          # HTTP 클라이언트 + LLM 체이닝
│   ├── pi0_ws_client.py             # WebSocket 스트리밍 클라이언트
│   ├── pi0_voice_client.py          # 음성 명령 클라이언트
│   ├── voice_module.py              # 음성 인식 모듈
│   └── chatbot_module.py            # 챗봇 모듈
├── dobot_vla/
│   ├── domain/                       # 순수 도메인 규칙 (pose/action/task)
│   ├── infrastructure/               # DOBOT, 카메라, Pi0 서버 adapter
│   └── application/                  # inference/planning use case
└── docs/
    ├── PIPELINE.md                  # 전체 파이프라인 가이드
    ├── team_guide.md                # 팀원용 복사-붙여넣기 실행 가이드
    ├── architecture_comparison.md   # 아키텍처 비교
    ├── ddd_modularization_guide.md  # DDD 모듈화 구현 설명
    ├── ddd_modularization_plan.md   # DDD 모듈화 계획
    ├── troubleshooting.md           # 트러블슈팅 모음
    ├── data_collection_guide.md     # 데이터 수집 가이드
    └── execution_plan.md            # 실행 계획
```

### 모듈화 구조

새 코드에서는 `dobot_vla` 패키지를 우선 사용합니다.

- `dobot_vla.domain`: 하드웨어와 무관한 pose/action/task 규칙
- `dobot_vla.infrastructure`: pydobot, OpenCV, HTTP, Pi0 runtime adapter
- `dobot_vla.application`: 카메라 → Pi0 → DOBOT 실행 흐름

기존 `client/`, `server/`, `scripts/` 파일은 실행용 entry point로 유지합니다.
자세한 기준은 [DDD 모듈화 계획](docs/ddd_modularization_plan.md)을 참고하세요.

---

## 사용법

처음 받는 팀원은 이 README를 먼저 보고, 실제 현장에서 그대로 따라 할 때는
[팀원용 실행 가이드](docs/team_guide.md)를 함께 열어두면 됩니다.

| 목적 | 문서 |
|---|---|
| 처음 실행하는 팀원용 전체 절차 | [docs/team_guide.md](docs/team_guide.md) |
| 데이터 수집만 자세히 보기 | [docs/data_collection_guide.md](docs/data_collection_guide.md) |
| 서버/학습/추론 파이프라인 상세 | [docs/PIPELINE.md](docs/PIPELINE.md) |
| 자주 났던 에러와 해결법 | [docs/troubleshooting.md](docs/troubleshooting.md) |
| DDD 모듈화 구조 설명 | [docs/ddd_modularization_guide.md](docs/ddd_modularization_guide.md) |

### 0. 프로젝트 받기

```bash
git clone https://github.com/wjdtjdwo0609-cyber/Dobot_VLM_VLA.git
cd Dobot_VLM_VLA
```

### 1. 사전 준비

하드웨어:

- DOBOT Magician 전원 어댑터와 USB 케이블 연결
- USB 카메라 2대 연결
- DOBOT 표시등이 초록불인지 확인
- 데이터 수집/추론 기준 카메라 매핑: `--cam1=wrist`, `--cam2=top`

Python 환경:

```bash
conda activate lerobot
pip install -r requirements.txt
```

`ModuleNotFoundError`가 나오면 같은 환경에서 `pip install -r requirements.txt`를 다시 실행하세요.

### 2. 카메라 번호 확인

카메라 번호가 헷갈리면 먼저 아래 명령으로 어떤 번호가 어떤 카메라인지 확인합니다.

```bash
python -c "
import cv2
for i in range(5):
    cap = cv2.VideoCapture(i)
    if cap.isOpened():
        ret, frame = cap.read()
        if ret:
            print(f'Camera {i}: OK ({frame.shape[1]}x{frame.shape[0]})')
            cv2.imshow(f'Camera {i}', frame)
        cap.release()
cv2.waitKey(3000)
cv2.destroyAllWindows()
"
```

확인한 번호를 데이터 수집/추론 명령의 `--cam1`, `--cam2`에 넣습니다.

### 3. 데이터 수집

```bash
python scripts/01_collect_data.py \
    --cam1 0 --cam2 1 \
    --task "pick up the tissue" \
    --save_dir ./tissue_dataset_v1
```

수집 프로그램이 켜지면 OpenCV 프리뷰 창을 클릭한 뒤 키를 누릅니다.

| 키 | 동작 |
|---|---|
| `S` | 현재 카메라 영상과 로봇 위치 캡처 |
| `E` | 손으로 이동한 뒤 delta action 기록 |
| `V` | 현재 에피소드 저장 |
| `D` | 현재 에피소드 버리기 |
| `W` | 마지막 스텝 되돌리기 |
| `G` | 그리퍼 ON/OFF |
| `Q` | 홈 위치로 이동 |
| `A` | 홈잉/캘리브레이션 |
| `X` | DOBOT 알람 해제 |
| `ESC` | 종료 |

한 에피소드는 보통 `S -> 팔 이동 -> E`를 여러 번 반복한 뒤 `V`로 저장합니다.
물체 하나당 최소 30개 이상의 성공 에피소드를 모으는 것을 권장합니다.

기존 데이터셋에 이어서 수집할 때는 반드시 `--resume`을 붙입니다.

```bash
python scripts/01_collect_data.py \
    --cam1 0 --cam2 1 \
    --task "pick up the tissue" \
    --save_dir ./tissue_dataset_v1 \
    --resume
```

`--resume` 없이 같은 `--save_dir`를 쓰면 기존 데이터가 새 수집으로 교체될 수 있습니다.

시리얼 포트가 자동 감지되지 않으면 직접 지정합니다.

```bash
python scripts/01_collect_data.py \
    --port /dev/tty.usbserial-XXXXX \
    --cam1 0 --cam2 1 \
    --task "pick up the tissue" \
    --save_dir ./tissue_dataset_v1
```

### 4. 데이터 검증

```bash
python scripts/03_validate_dataset.py --dataset_dir ./tissue_dataset_v1 --fix
```

검증은 학습 전에 반드시 실행합니다. `--fix`는 `tasks.parquet`, `meta/info.json` 등
복구 가능한 메타데이터 문제를 자동으로 고칩니다.

### 5. 서버로 데이터 보내기

```bash
scp -r ./tissue_dataset_v1 busan01@[서버IP]:~/snap/snapd-desktop-integration/intel_third_hands/Dobot_VLM_VLA/
```

예시:

```bash
scp -r ./tissue_dataset_v1 busan01@192.168.0.100:~/snap/snapd-desktop-integration/intel_third_hands/Dobot_VLM_VLA/
```

이미지 파일까지 함께 보내야 하므로 `scp -r`로 데이터셋 폴더 전체를 전송합니다.

### 6. 학습 (GPU 서버)

서버에 접속합니다.

```bash
ssh busan01@[서버IP]
cd ~/snap/snapd-desktop-integration/intel_third_hands/Dobot_VLM_VLA
source .venv/bin/activate
```

빠른 테스트:

```bash
./train.sh ./tissue_dataset_v1 1 100 outputs/tissue_test
```

본 학습:

```bash
./train.sh ./tissue_dataset_v1 1 10000 outputs/tissue_v1
```

중간에 멈춘 학습을 이어서 실행:

```bash
./train.sh ./tissue_dataset_v1 1 20000 outputs/tissue_v1 resume
```

여러 데이터셋을 합쳐서 학습:

```bash
./train.sh "./tissue_dataset_v1 ./cup_dataset_v1" 1 10000 outputs/multi_v1
```

인자 의미:

| 인자 | 예시 | 설명 |
|---|---|---|
| 데이터셋 경로 | `./tissue_dataset_v1` | 학습할 데이터셋 |
| GPU 번호 | `1` | `nvidia-smi`로 확인한 GPU 번호 |
| 학습 스텝 | `10000` | 학습 반복 횟수 |
| 출력 경로 | `outputs/tissue_v1` | 체크포인트 저장 위치 |
| resume | `resume` | 같은 출력 경로에서 이어서 학습 |

> `n_action_steps=1`: DOBOT의 `move_to(wait=True)`가 약 0.3~1.0초 걸리므로,
> 한 스텝만 실행하고 다시 관측하는 closed-loop 방식이 안정적입니다.

### 7. 추론 서버 실행

```bash
PI0_POLICY_TYPE=pi0_fast \
PI0_MODEL_PATH=./outputs/tissue_v1/checkpoints/last/pretrained_model \
python server/pi0_server.py
```

서버가 켜져 있는지 Mac에서 확인합니다.

```bash
curl http://[서버IP]:8000/health
```

`"status":"ok"`가 보이면 준비 완료입니다.

서버 LoRA 로딩 과정:

1. `lerobot/pi0fast-base` base 모델 로드
2. 체크포인트에서 LoRA 어댑터 merge
3. fine-tuned config 적용 (카메라: top + wrist, state: 5차원)

### 8. Mac에서 DOBOT 추론 실행

```bash
python client/pi0_dobot_client.py \
    --server http://[서버IP]:8000 \
    --cam1 0 --cam2 1 \
    --task "pick up the tissue" \
    --chunk-size 1
```

추론 클라이언트 키:

| 키 | 동작 |
|---|---|
| `SPACE` | 1회 추론 실행 |
| `A` | 자동 모드 ON/OFF |
| `H` | 홈 위치로 이동 |
| `G` | 그리퍼 ON/OFF |
| `T` | task 변경 |
| `L` | LLM 체이닝 모드 |
| `ESC` | 종료 |

처음에는 자동 모드보다 `SPACE`로 한 스텝씩 방향을 확인하는 것이 안전합니다.

### 9. 음성 명령 모드

```bash
python client/pi0_voice_client.py \
    --server http://[서버IP]:8000
```

음성 명령은 STT 결과를 task prompt로 변환한 뒤 같은 Pi0 추론 서버를 호출합니다.
물체명이나 명령어가 학습 때 사용한 문장과 너무 다르면 성능이 떨어질 수 있습니다.

---

## 모델 상세

| 항목 | 값 |
|------|-----|
| 모델 | Pi0-FAST (autoregressive + FAST tokenizer) |
| base 모델 | `lerobot/pi0fast-base` (PaliGemma-3B + Gemma-300M) |
| fine-tuning | LoRA adapter |
| 입력 이미지 | top + wrist 카메라 (480x640 → 224x224 자동 리사이즈) |
| 입력 state | `[x, y, z, r, gripper]` 5차원 (MEAN_STD 정규화) |
| 출력 action | `[Δx, Δy, Δz, Δr, grip]` 5차원 (delta) |
| chunk_size | 5 |

---

## 하드웨어 예시

| 구성 | 사양 |
|------|------|
| 로봇 | DOBOT Magician (USB, CH340/CP210x) |
| 카메라 | USB 카메라 2대 (cam1=wrist, cam2=top), 640x480 |
| 학습 GPU | A6000+ (48GB VRAM) |

---

## 라이센스 정책
See [LICENSE](./LICENSE)
