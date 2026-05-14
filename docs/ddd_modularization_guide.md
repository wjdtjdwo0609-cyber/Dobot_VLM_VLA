# DDD Modularization Guide

이 문서는 VLA 코드를 어떤 기준으로 모듈화했는지 설명하는 개발자용 가이드입니다.
계획 중심 문서는 `docs/ddd_modularization_plan.md`이고, 이 문서는 실제 코드 구조를 읽는 순서에 맞췄습니다.

## 한 줄 요약

기존에는 `client/`, `server/`, `scripts/` 실행 파일이 로봇 제어, 카메라, Pi0 통신, 태스크 해석, 실행 루프를 직접 들고 있었습니다.
지금은 실행 파일은 진입점으로 남기고, 재사용 가능한 코드는 `dobot_vla/` 패키지로 분리했습니다.

```
Presentation  ->  Application  ->  Domain
      |                 |
      v                 v
Infrastructure adapters  Domain rules
```

## 모듈화 기준

DDD 관점에서 “바뀌는 이유”가 다른 코드를 분리했습니다.

| 경계 | 바뀌는 이유 | 예시 |
|---|---|---|
| Robot Control | 로봇 좌표계, 안전 범위, 그리퍼 타이밍 변경 | DOBOT에서 다른 로봇 팔로 변경 |
| Vision Capture | 카메라 번호, 백엔드, 인코딩 방식 변경 | USB 카메라에서 RTSP/ZMQ로 변경 |
| VLA Inference | Pi0 서버, HTTP/WS, 정규화, 모델 로딩 변경 | Pi0-FAST에서 다른 policy로 변경 |
| Task Routing | 한국어 alias, 학습 프롬프트, LLM planner 변경 | 새 물체/명령 추가 |
| Application Flow | 실행 순서와 UI 조작 변경 | 수동 모드, 자동 모드, 음성 모드 |

## 현재 폴더 구조

```
dobot_vla/
├── domain/
│   ├── robot.py
│   └── tasks.py
├── infrastructure/
│   ├── camera.py
│   ├── dobot.py
│   ├── pi0_client.py
│   └── pi0_runtime.py
└── application/
    ├── inference_service.py
    └── planning.py
```

## Domain 계층

Domain은 프로젝트의 핵심 규칙입니다. 여기에는 `cv2`, `pydobot`, `torch`, `requests` 같은 외부 의존성을 넣지 않습니다.

### `dobot_vla/domain/robot.py`

역할:

- `RobotPose`: DOBOT pose `x, y, z, r`
- `RobotState`: Pi0 입력 형태 `[x, y, z, r, grip]`
- `DeltaAction`: Pi0 출력 형태 `[dx, dy, dz, dr, grip]`
- `SafetyBounds`: 로봇 작업 공간 제한

핵심 규칙:

```text
현재 pose + Pi0 delta action -> target pose -> safety bounds clamp -> robot command
```

즉, 모델이 큰 delta를 내더라도 최종 target pose는 정해진 작업 공간 안으로 제한됩니다.

### `dobot_vla/domain/tasks.py`

역할:

- 한국어 물체명과 alias를 Pi0가 학습한 영어 prompt로 매핑합니다.
- 예: `휴지`, `휴지 좀` -> `pick up the tissue and hand it over`

중요한 점:

- Pi0는 학습 때 사용한 prompt에 민감하므로, 이 문자열은 단순 번역문이 아니라 모델 조건화 규칙입니다.
- 새 물체를 추가할 때는 여기의 `COMMAND_MAP`에 alias와 prompt를 추가합니다.

## Infrastructure 계층

Infrastructure는 외부 장치와 라이브러리를 감싸는 adapter입니다. 도메인 규칙을 새로 만들지 않고, Domain 객체를 받아 실제 외부 세계에 연결합니다.

### `dobot_vla/infrastructure/dobot.py`

역할:

- `pydobot` 연결
- 포트 자동 감지
- pose/state 읽기
- delta action 실행
- gripper, home, homing

중요한 점:

- Application은 `pydobot`을 직접 만지지 않고 `DobotGateway`만 사용합니다.
- `DobotGateway.execute()`는 `DeltaAction`을 받고, Domain의 `SafetyBounds`로 target을 clamp한 뒤 실제 `move_to()`를 호출합니다.
- 그리퍼 타이밍은 `DobotConfig.gripper_delay_s`, `DobotConfig.gripper_wait_s`에서 조정합니다.

### `dobot_vla/infrastructure/camera.py`

역할:

- OpenCV 카메라 열기
- 현재 기준 카메라 매핑 통일: `cam1=wrist`, `cam2=top`
- Pi0 요청용 JPEG/base64 인코딩

이 모듈 덕분에 HTTP 클라이언트와 WebSocket 클라이언트가 같은 카메라 규칙을 씁니다.

### `dobot_vla/infrastructure/pi0_client.py`

역할:

- Pi0 HTTP 서버 health check
- 카메라 프레임을 JPEG/base64 payload로 변환
- `/predict` 요청
- 기존 코드와 호환되는 `(actions, raw_actions, inference_time_ms)` 반환

### `dobot_vla/infrastructure/pi0_runtime.py`

역할:

- Pi0/Pi0-FAST 모델 로딩
- LoRA adapter merge
- LeRobot normalizer 로딩
- 이미지/state/language observation 구성
- action 역정규화

중요한 점:

- `server/pi0_server.py`와 `server/pi0_ws_server.py`가 같은 runtime을 공유합니다.
- HTTP와 WebSocket은 transport만 다르고 모델 추론 로직은 동일합니다.

## Application 계층

Application은 “무슨 순서로 실행할지”만 조립합니다.

### `dobot_vla/application/inference_service.py`

역할:

```text
camera.capture()
-> dobot.get_state()
-> pi0.predict(image, state, task)
-> dobot.execute(delta)
```

여기에는 수동 모드, 자동 모드, LLM 체이닝 실행 흐름이 들어 있습니다.
다만 실제 카메라, 실제 DOBOT, 실제 HTTP 요청의 세부 구현은 Infrastructure adapter에 맡깁니다.

### `dobot_vla/application/planning.py`

역할:

- 고수준 목표를 여러 개의 Pi0 task prompt로 분해합니다.
- `simple`, `local`, `openai`, `anthropic` backend를 지원합니다.

## Presentation 계층

기존 실행 파일은 사용자가 실행하는 entry point로 유지했습니다.

| 파일 | 지금 역할 |
|---|---|
| `client/pi0_dobot_client.py` | argparse 후 `RemoteInferencePipeline` 조립 |
| `client/pi0_ws_client.py` | WebSocket transport용 CLI |
| `client/pi0_voice_client.py` | STT + chatbot + Pi0 실행 조립 |
| `server/pi0_server.py` | FastAPI HTTP endpoint |
| `server/pi0_ws_server.py` | FastAPI WebSocket endpoint |

즉, 기존 명령어는 유지하되 내부 로직은 `dobot_vla`로 이동했습니다.

## 기존 코드에서 이동한 책임

| 이전 위치 | 이동한 책임 | 새 위치 |
|---|---|---|
| `client/pi0_dobot_client.py` | HTTP Pi0 요청 | `infrastructure/pi0_client.py` |
| `client/pi0_dobot_client.py` | DOBOT 실행/안전 clamp | `infrastructure/dobot.py`, `domain/robot.py` |
| `client/pi0_dobot_client.py` | 카메라 캡처 | `infrastructure/camera.py` |
| `client/pi0_dobot_client.py` | LLM planner | `application/planning.py` |
| `server/pi0_server.py` | 모델 로딩/정규화/추론 | `infrastructure/pi0_runtime.py` |
| `server/pi0_ws_server.py` | 모델 로딩/정규화/추론 | `infrastructure/pi0_runtime.py` |
| `client/chatbot_module.py` | command map/stop keywords | `domain/tasks.py` |

## 코드 주석 기준

각 모듈에는 다음 방식으로 설명을 붙였습니다.

- 파일 상단 docstring: 이 모듈이 어느 계층이고 왜 존재하는지 설명
- 클래스 docstring: 외부에서 이 클래스를 어떤 의도로 써야 하는지 설명
- 위험하거나 헷갈리는 흐름의 인라인 주석: safety clamp, 카메라 순서, Pi0 payload, gripper timing

예를 들어 `DobotGateway.execute()`는 모델 delta를 바로 실행하지 않고:

1. `DeltaAction`으로 변환
2. 현재 pose 읽기
3. target pose 계산
4. safety bounds clamp
5. `move_to()` 실행
6. gripper 상태 업데이트

순서로 동작하도록 주석과 코드 구조를 맞췄습니다.

## 테스트

하드웨어 없이 검증 가능한 최소 도메인 테스트를 추가했습니다.

```bash
python -m unittest discover -s tests
```

현재 테스트 범위:

- delta action이 safety bounds 안으로 clamp되는지
- `RobotState`가 Pi0 입력 shape로 변환되는지
- 한국어 object alias가 학습 prompt로 매핑되는지

## 다음 리팩터링 후보

아직 가장 큰 파일은 `scripts/01_collect_data.py`입니다.
다음 단계에서는 이 파일을 아래처럼 나누는 게 좋습니다.

```text
application/data_collection_service.py
infrastructure/lerobot_dataset_writer.py
infrastructure/collector_preview.py
domain/dataset_schema.py
```

이렇게 하면 데이터 수집, 검증, 복구 로직도 현재 inference 쪽과 같은 구조로 정리됩니다.
