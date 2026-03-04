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

## 6) `mask_action` 동작 정리 (미사일 회피)

아래 내용은 `SingleCombatDodgeMissileTask.mask_action()`의 현재 로직을 기준으로 합니다.

### 6.1 미사일 접근 방향 기반 회피 방향

- 미사일 경보 시 2D 속도벡터(ego/missile)만 사용해 헤딩을 계산합니다.
- 미사일 진행방향에 대해 수직인 두 후보 헤딩(좌/우)을 만들고,
  ego가 더 적게 꺾는 쪽을 목표 회피 방향으로 선택합니다.

### 6.2 후보 action 마스킹 (7개 중 5개 허용)

- 미사일 경보 시 회피 후보 action 7개를 사용합니다.
- 목표가 좌선회면 우선회 계열 2개를 mask,
  목표가 우선회면 좌선회 계열 2개를 mask 합니다.
- 따라서 최종적으로 **5개 action만 선택 가능**합니다.

### 6.3 점수 벡터 입력 시 선택 규칙

- 입력 action이 7차원 점수/로짓 벡터이면,
  마스킹된 인덱스 점수를 `-inf`로 만든 뒤 `argmax`를 선택합니다.
- 선택된 인덱스의 action 템플릿(저수준 4차원)을 반환합니다.

### 6.4 속도 고려

- 미사일 속도가 빠른 경우(코드 기준 600 m/s 이상),
  저수준 회피 action의 elevator를 추가로 키워 선회를 강화합니다.


## 7) Reward 계산 수식 정리 (JSBSim)

아래 수식은 `external/CloseAirCombat/envs/JSBSim/reward_functions/`의 구현을 기준으로 정리했습니다.

### 7.1 공통 처리 (`BaseRewardFunction`)

각 reward 함수가 계산한 원시 보상 \(r_t^{raw}\)에 대해:

\[
r_t^{scaled} = s \cdot r_t^{raw}
\]

- \(s\): `<RewardClassName>_scale`

Potential-based shaping이 켜져 있으면(`..._potential=true`):

\[
r_t = r_t^{scaled} - r_{t-1}^{scaled}
\]

아니면:

\[
r_t = r_t^{scaled}
\]

---

### 7.2 `PostureReward`

적기 대비 AO, TA, 거리 \(R\)를 이용해:

\[
r_{posture} = f_{orn}(AO, TA) \cdot f_{range}(R_{km})
\]

코드 기본값은 `orientation_version=v2`, `range_version=v3`입니다.

#### Orientation (v2)
\[
f_{orn}^{v2}(AO,TA)=\frac{1}{50AO/\pi+2}+\frac{1}{2}+\min\left(\frac{\operatorname{arctanh}(1-\max(2TA/\pi,10^{-4}))}{2\pi},0\right)+0.5
\]

#### Range (v3)
\[
f_{range}^{v3}(R)=\mathbf{1}_{R<5}+\mathbf{1}_{R\ge5}\cdot\text{clip}(-0.032R^2+0.284R+0.38,0,1)+\text{clip}(e^{-0.16R},0,0.2)
\]

여기서 \(R\) 단위는 km입니다.

---

### 7.3 `AltitudeReward`

- 자 기체 고도: \(z\) (km)
- 자 기체 수직속도: \(v_z\) (마하 단위 정규화, 코드: m/s ÷ 340)

속도 페널티:
\[
P_v=
\begin{cases}
-\text{clip}\left(\frac{v_z}{K_v}\cdot\frac{H_{safe}-z}{H_{safe}},0,1\right), & z\le H_{safe}\\
0, & z>H_{safe}
\end{cases}
\]

고도 페널티:
\[
P_H=
\begin{cases}
\text{clip}\left(\frac{z}{H_{danger}},0,1\right)-2, & z\le H_{danger}\\
0, & z>H_{danger}
\end{cases}
\]

총 보상:
\[
r_{alt}=P_v+P_H
\]

---

### 7.4 `EventDrivenReward`

이산 이벤트 보상:

\[
r_{event}=
\begin{cases}
-200, & \text{is\_shotdown}\\
-200, & \text{is\_crash}\\
0, & \text{otherwise}
\end{cases}
+100\times N_{launch}^{new}
+100\times N_{enemy\_shotdown}^{new}
\]

- \(N_{launch}^{new}\): 현재 스텝에서 새롭게 발생한 미사일 발사 개수
- \(N_{enemy\_shotdown}^{new}\): 현재 스텝에서 새롭게 발생한 적 기체 격추 수

---

### 7.5 `MissilePostureReward`

미사일 경보가 있을 때:

\[
\Delta v_m = \frac{\|v_{m,t-1}\|-\|v_{m,t}\|}{340}\cdot s
\]
\[
a = \frac{v_m\cdot v_e}{\|v_m\|\|v_e\|}
\]

보상은 다음과 같이 계산:

\[
r_{missile}=
\begin{cases}
\dfrac{a}{\max(\Delta v_m,0)+1}, & a<0\\
a\cdot\max(\Delta v_m,0), & a\ge0
\end{cases}
\]

미사일이 없으면 \(r_{missile}=0\).

> 참고: 이 클래스는 `_process()`를 거치지 않고 직접 `reward`를 반환합니다.

---

### 7.6 `ShootPenaltyReward`

스텝 사이에 잔여 미사일이 1개 감소(= 발사)하면:

\[
r_{shoot}=-10
\]

아니면:
\[
r_{shoot}=0
\]

---

### 7.7 기타 reward 함수

#### `HeadingReward`
각 오차 항에 대한 가우시안 보상:
\[
r_h=e^{-(\Delta\psi/5)^2},\quad
r_{alt}=e^{-(\Delta h/15.24)^2},\quad
r_{roll}=e^{-(\phi/0.35)^2},\quad
r_v=e^{-(\Delta u/24)^2}
\]

최종 보상(기하평균):
\[
r_{heading}=(r_h\,r_{alt}\,r_{roll}\,r_v)^{1/4}
\]

#### `RelativeAltitudeReward`
\[
r_{rel\_alt}=\min\left(K_H-|z_{ego}-z_{enm}|,\ 0\right)
\]

---

### 7.8 Task별 reward 합산

Task는 활성화된 reward 함수들을 단순 합산합니다.

\[
r_{total}=\sum_i r_i
\]

`SingleCombatTask` 기본 구성:
- `AltitudeReward`
- `PostureReward`
- `EventDrivenReward`

`SingleCombatDodgeMissileTask`:
- `PostureReward`
- `MissilePostureReward`
- `AltitudeReward`
- `EventDrivenReward`

`SingleCombatShootMissileTask`:
- `PostureReward`
- `AltitudeReward`
- `EventDrivenReward`
- `ShootPenaltyReward`
