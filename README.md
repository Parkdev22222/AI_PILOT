# CloseAirCombat 2v2 + CLAM Training Integration

이 레포는 **CloseAirCombat 2vs2 공중교전 환경**에서 **CLAM 알고리즘**으로 학습을 실행하기 위한 통합 스캐폴딩을 제공합니다.

> ⚠️ 현재 실행 환경에서는 GitHub 접근이 차단되어 원본 레포를 자동으로 가져올 수 없습니다.
> 아래 절차대로 두 레포를 수동으로 내려받은 뒤 실행하세요.

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
  --clam-entry "clam_closeaircombat.clam_trainer:CLAMTrainer" \
  --total-steps 500000
```

> 실행 시 `ModuleNotFoundError: No module named 'clam_closeaircombat'`가 발생한다면,
> `scripts/train_clam_2v2.py`가 자동으로 `src/`를 `PYTHONPATH`에 추가하도록 되어 있으니
> 최신 변경사항을 받아주세요.

### 참고
- `--env-entry`는 `module_path:callable_name` 형식입니다.
- 기본 예시는 `closeaircombat.envs:make_2v2_env`이며, 내부에서 JSBSim 환경을 생성하고
  CLAM이 사용할 수 있도록 **MultiDiscrete 액션을 단일 Discrete로 펼치는 어댑터**를 적용합니다.
  `env_kwargs.config_name`으로 `2v2/NoWeapon/HierarchySelfplay` 같은 JSBSim config 이름을 지정할 수 있습니다.
- `--clam-entry`는 CLAM 트레이너 클래스 경로이며, 기본 제공되는
  `clam_closeaircombat.clam_trainer:CLAMTrainer`는 CLAM-RL의 PPO+FNN 모델을 사용해
  CloseAirCombat 환경에서 학습하도록 구성되어 있습니다.

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
  clam_trainer.py
  config.py
  runner.py
  utils.py
./src/closeaircombat/
  envs.py
```
