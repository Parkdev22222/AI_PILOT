# CloseAirCombat 2v2 + CLAM Training Integration

이 레포는 **CloseAirCombat 2vs2 공중교전 환경**에서 **CLAM 알고리즘**으로 학습을 실행하기 위한 통합 스캐폴딩을 제공합니다.

> ⚠️ 현재 실행 환경에서는 GitHub 접근이 차단되어 원본 레포를 자동으로 가져올 수 없습니다.
> 아래 절차대로 두 레포를 수동으로 내려받은 뒤 실행하세요.

---

## 전술 시뮬레이터 UI 실행

### 로컬 실행

```bash
# 1. 라이브러리 설치
pip install -r requirements.txt

# 2. UI 실행 (기본 포트 7860)
python -m tactical_system.run_with_ui

# 주요 옵션
python -m tactical_system.run_with_ui \
    --ego_policy  path/to/ego_actor.pt \       # 아군 RL 정책 체크포인트
    --enm_policy  path/to/enm_actor.pt \       # 적군 RL 정책 체크포인트
    --llm_model   LGAI-EXAONE/EXAONE-3.5-7.8B-Instruct \  # HuggingFace 모델
    --db_path     combat.db \                  # SQLite DB 경로
    --num_enemy   2 \                          # 적 편대 수 (미입력 시 1~3 랜덤)
    --port        7860 \                       # Gradio 포트
    --share                                    # 외부 접속용 공개 링크 생성
```

브라우저에서 `http://localhost:7860` 접속.

---

### Google Colab 실행

```python
# 1. 레포 클론
!git clone <레포_URL>
%cd AI_PILOT

# 2. 외부 레포 준비
!mkdir -p external
!git clone https://github.com/snu-larr/CloseAirCombat.git external/CloseAirCombat
!git clone https://github.com/WenhaoMa-UTS/CLAM-RL.git external/CLAM-RL

# 3. 의존 라이브러리 설치
!pip install -r requirements.txt

# 4. UI 실행 (share=True → 외부 접속 URL 자동 발급)
!python -m tactical_system.run_with_ui \
    --llm_model LGAI-EXAONE/EXAONE-3.5-7.8B-Instruct \
    --db_path   combat.db \
    --port      7860 \
    --share
```

또는 Python 코드에서 직접 실행:

```python
import sys
sys.path.insert(0, "/content/AI_PILOT")

from tactical_system.gradio_dashboard import TacticalDashboard

dashboard = TacticalDashboard(
    db_path="combat.db",
    refresh_interval=2.0,
)
dashboard.launch(server_port=7860, share=True)
```

> **Colab 팁**
> - GPU 런타임 권장 (`런타임 > 런타임 유형 변경 > GPU`)
> - `--share` 플래그 없이 실행하면 Colab 외부에서 접속 불가
> - LLM 로드는 수십 초 소요 — 셀 실행 후 `[시뮬레이션 초기화 완료]` 로그 확인 후 UI 사용

---

### UI 사용 방법

1. **`⚙ 시나리오 설정`** 버튼 클릭 → 모달 팝업
2. **적군 출격 기지** 체크박스로 복수 선택
3. **지도 클릭** → 클릭 위치에 ★ 마커 표시 → 공격 목표 자동 설정
   또는 **미리 정의된 목표** 라디오 버튼으로 프리셋 선택
4. **`▶ 시뮬레이션 시작`** 클릭 → 실시간 전황 대시보드로 전환

---

## 1) 외부 레포 준비

```bash
mkdir -p external
cd external
# CloseAirCombat
git clone https://github.com/snu-larr/CloseAirCombat.git
# CLAM-RL
git clone https://github.com/WenhaoMa-UTS/CLAM-RL.git
```

## 2) 환경 변수(선택)

레포 경로를 명시적으로 지정하려면 아래 환경 변수를 설정합니다.

```bash
export CLOSE_AIR_COMBAT_PATH="$(pwd)/external/CloseAirCombat"
export CLAM_RL_PATH="$(pwd)/external/CLAM-RL"
```

## 3) 학습 실행

아래 스크립트는 **환경 생성 함수**와 **CLAM 진입점**을 문자열로 받습니다.

```bash
python scripts/train_clam_2v2.py \
  --env-entry "closeaircombat.envs:make_2v2_env" \
  --clam-entry "clam_rl.trainers.clam_trainer:CLAMTrainer" \
  --total-steps 500000
```

> 실행 시 `ModuleNotFoundError: No module named 'clam_closeaircombat'`가 발생한다면,
> `scripts/train_clam_2v2.py`가 자동으로 `src/`를 `PYTHONPATH`에 추가하도록 되어 있으니
> 최신 변경사항을 받아주세요.

### 참고
- `--env-entry`는 `module_path:callable_name` 형식입니다.
- 예: `CloseAirCombat.envs.JSBSim.envs.multiplecombat_env:MultipleCombatEnv`처럼
  레포 이름으로 시작하는 모듈 경로도 사용할 수 있도록 `sys.path`에 레포 부모 경로를 추가합니다.
  예를 들어 CloseAirCombat이 `external/CloseAirCombat`에 있다면
  `external`이 자동으로 `sys.path`에 포함되어 위 모듈 경로가 임포트됩니다.
- `--clam-entry`는 CLAM-RL에서 **실제 trainer/agent 클래스**의 경로로 교체해야 합니다.
  예: `clam_rl.trainers.clam_trainer:CLAMTrainer` 또는
  `clam_rl.agents.clam_agent:CLAMAgent` (프로젝트 구조에 맞게 조정)
- 위 entry는 예시이며, 실제 모듈/클래스 경로는 두 레포의 구조에 맞게 변경하세요.

## 4) 추가 설정

`--config` 옵션으로 YAML/JSON 설정 파일을 전달할 수 있습니다.

```bash
python scripts/train_clam_2v2.py --config configs/clam_closeaircombat.yaml
```

## 5) 에러 검증

통합 스크립트는 다음을 확인합니다:
- 레포 경로 유효성 (CloseAirCombat, CLAM-RL)
- 엔트리 포인트 유효성 (import 가능 여부)
- 환경/알고리즘 생성 실패 시 상세 오류 출력

---

### 파일 구조

```
./scripts/train_clam_2v2.py
./src/clam_closeaircombat/
  env_adapter.py
  clam_adapter.py
  config.py
  runner.py
  utils.py
```
