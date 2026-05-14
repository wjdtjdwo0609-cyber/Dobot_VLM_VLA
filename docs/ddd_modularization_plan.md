# DDD 모듈화 계획

이 문서는 기존 `scripts/`, `client/`, `server/` 중심의 VLA 코드를 새 팀원이 이해하고 재사용하기 쉬운 구조로 나누기 위한 기준입니다.
실제 코드가 어떻게 나뉘었는지 파일별로 읽고 싶다면 `docs/ddd_modularization_guide.md`를 먼저 보면 됩니다.

## 목표

- 로봇 제어, 카메라, Pi0 통신, 태스크 라우팅을 복사-붙여넣기 없이 재사용한다.
- 하드웨어/API 의존 코드를 도메인 규칙과 분리한다.
- 기존 실행 명령은 유지한다.
- 위험한 로봇 동작 규칙은 한 곳에서만 관리한다.

## 현재 적용된 구조

```
dobot_vla/
├── domain/
│   ├── robot.py          # RobotPose, RobotState, DeltaAction, SafetyBounds
│   └── tasks.py          # 한국어 물체명 -> Pi0 영어 프롬프트 카탈로그
├── infrastructure/
│   ├── camera.py         # cam1=wrist, cam2=top 규칙과 JPEG 인코딩
│   ├── dobot.py          # pydobot 연결, homing, gripper, safety clipping
│   ├── pi0_client.py     # HTTP Pi0 서버 클라이언트
│   └── pi0_runtime.py    # 서버 공통 모델 로딩/정규화/추론 런타임
└── application/
    ├── inference_service.py  # 카메라 -> Pi0 -> DOBOT 실행 루프
    └── planning.py           # 선택적 LLM planner
```

기존 CLI 파일은 Presentation 계층으로 남긴다.

```
client/pi0_dobot_client.py  # HTTP CLI
client/pi0_ws_client.py     # WebSocket CLI
server/pi0_server.py        # HTTP FastAPI transport
server/pi0_ws_server.py     # WebSocket transport
```

## Bounded Context

| Context | 역할 | 대표 파일 |
|---|---|---|
| Robot Control | DOBOT pose/state/action과 안전 범위 | `domain/robot.py`, `infrastructure/dobot.py` |
| Vision Capture | wrist/top 카메라 순서, 캡처, 인코딩 | `infrastructure/camera.py` |
| VLA Inference | Pi0 요청/응답, 서버 모델 런타임 | `infrastructure/pi0_client.py`, `infrastructure/pi0_runtime.py` |
| Task Routing | 한국어 입력/물체명에서 Pi0 prompt 생성 | `domain/tasks.py`, `application/planning.py` |
| Application Flow | closed-loop inference orchestration | `application/inference_service.py` |

## 적용 완료

- `client/pi0_dobot_client.py`의 카메라, DOBOT, Pi0 HTTP, LLM planner 책임을 모듈로 분리했다.
- `client/pi0_ws_client.py`가 같은 카메라/DOBOT 모듈을 사용하도록 바꿨다.
- `server/pi0_server.py`와 `server/pi0_ws_server.py`가 같은 Pi0 runtime을 사용하도록 통합했다.
- `client/chatbot_module.py`의 COMMAND_MAP/STOP_KEYWORDS를 `domain/tasks.py`로 옮겼다.
- 기존 실행 명령과 주요 클래스명(`Pi0Client`, `DobotController`, `CameraManager`)은 호환 alias로 유지했다.

## 다음 단계

1. `scripts/01_collect_data.py`를 `data_collection` application service와 `LeRobotV3DatasetWriter` 인프라 모듈로 나눈다.
2. `scripts/03_validate_dataset.py` 검증 규칙을 도메인 규칙으로 분리하고 테스트를 붙인다.
3. 하드웨어 없이 돌릴 수 있는 fake camera/fake dobot adapter를 추가한다.
4. 실제 DOBOT 연결 전에도 `pytest`로 safety clipping, task routing, payload serialization을 검증한다.

## 작업 원칙

- Domain 계층은 `cv2`, `pydobot`, `torch`, `requests`를 import하지 않는다.
- Infrastructure 계층은 외부 라이브러리를 감싸지만, 도메인 규칙을 새로 만들지 않는다.
- Application 계층은 순서를 조립한다. 하드웨어 세부 구현은 adapter에 맡긴다.
- Presentation 계층은 argparse/FastAPI/WebSocket 같은 입출력만 담당한다.
