# Light Aircraft Game: A lightweight, scalable, gym-wrapped aircraft competitive environment with baseline reinforcement learning algorithms
We provide a competitive environment for red and blue aircrafts games, which includes single control setting, 1v1 setting and 2v2 setting. The flight dynamics based on JSBSIM, and missile dynamics based on our implementation of proportional guidance. We also provide ppo and mappo implementation for self-play or vs-baseline training. 

![fromework](assets/framework.jpg)

## Install (update)
```bash
# Create conda env
conda create -n jsbsim python=3.8
conda activate jsbsim
# Install pytorch (gpu version)
conda install pytorch torchvision torchaudio pytorch-cuda=11.8 -c pytorch -c nvidia
# Install dependencies
pip install pymap3d jsbsim==1.1.6 gymnasium==0.26.2 shapely==1.7.1 geographiclib wandb icecream setproctitle
# Initialize submodule (JSBSim repo)
git submodule init; git submodule update
```
## Install (original)

```shell
# create python env
conda create -n jsbsim python=3.8
# install dependency
pip install torch pymap3d jsbsim==1.1.6 geographiclib gym==0.20.0 wandb icecream setproctitle. 

- Download Shapely‑1.7.1‑cp38‑cp38‑win_amd64.whl from [here](https://www.lfd.uci.edu/~gohlke/pythonlibs/#shapely), and `pip install shaply` from local file.

- Initialize submodules(*JSBSim-Team/jsbsim*): `git submodule init; git submodule update`
```
## Envs
We provide all task configs in  `envs/JSBSim/configs`, each config corresponds to a task.

### SingleControl
SingleControl env includes single agent heading task, whose goal is to train agent fly according to the given direction, altitude and velocity. The trained agent can be used to design baselines or become the low level policy of the following combat tasks. We can designed two baselines, as shown in the video:

![singlecontrol](assets/1_control.gif)

The red is manever_agent, flying in a triangular trajectory. The blue is pursue agent, constantly tracking the red agent. You can reproduce this by `python envs/JSBSim/test/test_baseline_use_env.py`.


### SingleCombat
SingleCombat env is for two agents 1v1 competitive tasks, including NoWeapon tasks and Missile tasks. We provide self-play setting and vs-baseline setting for each task. Due to the fact that learning to fly and combat simultaneously is non-trival, we also provide a hierarchical framework, where the upper level control gives the direction, altitude and velocity, the low level control use the model trained in SingleControl. 


- NoWeapon tasks require the agent to be in an posture advantage, which means the agent need to fly towards the tail of its opponent and maintain a proper distance. 
- Missile tasks require the agent learn to shoot down oppoents and dodge missiles. Missile engines are based on proportional guidance, we provide a document for our impletation [here](docs/missile_engine). We can futher divide missile tasks into into two categories:
  - Dodge missile task. Missile launches are controled by rules, train agent learn to dodge missile.
  - Shoot missile task. Missile launches are also learning goals. But training from scratch to learn launching missiles is not trival, we need to introduce some prior knowledge for policy learning. We use property that conjugate prior of binomial distribution is beta distribution to address this issue, refer to [here](docs/parameterized_shooting.md) for more details.  A demo for shoot missile task:

![1v1_missile](assets/1v1_missile.gif)


### MultiCombat
MultiCombat env is for four agents 2v2 competitive tasks. The setting is same as SingleCombat. A demo for non-weapon tasks: 

![2v2_posture](assets/2v2_posture.gif)

## Quick Start
### Training

```bash
cd scripts
bash train_*.sh
```
We have provide scripts for five tasks in `scripts/`.

- `train_heading.sh` is for SingleControl environment heading task.
- `train_vsbaseline.sh` is for SingleCombat vs-baseline tasks.
- `train_selfplay.sh` is for SingleCombat self-play tasks. 
- `train_selfplay_shoot.sh` is for SingleCombat self-play shoot missile tasks.
- `train_share_selfplay.sh` is for MultipleCombat self-play tasks.

It can be adapted to other tasks by modifying a few parameter settings. 

- `--env-name` includes options ['SingleControl', 'SingleCombat', 'MultipleCombat'].
- `--scenario` corresponds to yaml file in `envs/JBSim/configs` one by one.
- `--algorithm` includes options [ppo, mappo], ppo for SingleControl and SingleCombat, mappo for MultipleCombat

The description of parameter setting refers to `config.py`.
Note that we set parameters `--use-selfplay --selfplay-algorithm --n-choose-opponents --use-eval --n-eval-rollout-threads --eval-interval --eval-episodes` in selfplay-setting training. `--use-prior` is only set true for shoot missile tasks.
We use wandb to track the training process. If you set `--use-wandb`, please replace the `--wandb-name` with your name. 

### Evaluate and Render
```bash
cd renders
python render*.py
```
This will generate a `*.acmi` file. We can use [**TacView**](https://www.tacview.net/), a universal flight analysis tool, to open the file and watch the render videos.

## Citing
If you find this repo useful, pleased use the following citation:
````
@misc{liu2022light,
  author = {Qihan Liu and Yuhua Jiang and Xiaoteng Ma},
  title = {Light Aircraft Game: A lightweight, scalable, gym-wrapped aircraft competitive environment with baseline reinforcement learning algorithms},
  year = {2022},
  publisher = {GitHub},
  journal = {GitHub repository},
  howpublished = {\url{https://github.com/liuqh16/CloseAirCombat}},
}


## Air Commander LLM System (New)
- `command/air_commander_system.py`: 엑사원4 기반 공군 지휘관 에이전트 + 다중 기지 출격/다중 지역 교전 생성 오케스트레이터
- `command/commander_db.py`: 실시간 교전 DB 저장 모듈 (환경 ID, 교전 지역, 아군/적군 위치, 격추 여부, 잔여 무장량)
- `scripts/command/run_air_commander_system.py`: 다방향 적기 남하 시나리오를 실행하고 거리 40km 이내 시 `MultipleCombatEnv`를 생성
- `scripts/command/query_commander_rag.py`: EXAONE4 에이전트가 DB 조회 TOOL(RAG)로 상황 질의
- `scripts/command/run_commander_web.py`: **Gradio** 기반 한반도 지도 대시보드 서버 (지도 + 우측 이벤트 패널)
- `web/command_dashboard/index.html`: 교전 지역(투명 빨강) + 이동중 편대(아군/적군 원형) 실시간 시각화 페이지

시스템 개요:
1) LLM이 한반도 전역 적기 남하 경로를 보고 출격 기지를 결정
2) 아군/적군이 시속 2000km로 접근
3) 거리 40km 이내에서 지역별 `MultipleCombatEnv`가 동시 생성
4) 교전 중 이벤트(아군 격추/무장 고갈) 발생 시 LLM이 RTB 여부를 판단
5) 각 교전 환경 ID/지역/상태를 DB에 실시간 누적하고 LLM이 TOOL로 조회
6) 웹 대시보드에서 교전중 지역은 투명 빨강으로, 이동중 편대는 아군/적군 원형으로 실시간 표시


### Air Commander 실행 방법
1. 지휘 시뮬레이션 실행 (DB 생성 + run_id 출력)
```bash
cd external/CloseAirCombat
python scripts/command/run_air_commander_system.py \
  --db-path /tmp/commander_live.db \
  --scenario-name "2v2/NoWeapon/HierarchySelfplay" \
  --steps 300 \
  --dt-seconds 10 \
  --ego-policy-dir /path/to/ego_policy_dir \
  --enm-policy-dir /path/to/enm_policy_dir \
  --ego-policy-index latest \
  --enm-policy-index latest \
  --policy-device cpu \
  --local-exaone-path /path/to/local/exaone4 \
  --local-exaone-device cpu
```
- 정책 경로를 주지 않으면 교전 단계는 기본 fallback action으로 동작합니다.
- 실행 로그에서 `run_id=...` 값을 확인합니다.
- 로컬 EXAONE4를 사용하려면 `transformers` 설치 후 `--local-exaone-path`에 로컬 모델 경로를 지정합니다.

2. 실시간 지도 웹 대시보드(Gradio) 실행
```bash
cd external/CloseAirCombat
python scripts/command/run_commander_web.py \
  --db-path /tmp/commander_live.db \
  --run-id <위에서 출력된 run_id> \
  --host 0.0.0.0 \
  --port 8088
```
- 브라우저에서 `http://localhost:8088` 접속
- 교전중 지역: 투명 빨강 오버레이
- 이동중 편대: 아군/적군 원형 마커
- 지도 오른쪽 이벤트 패널: 교전 발생, 아군 격추, 적군 격추, LLM 아군 복귀 명령

3. EXAONE4 RAG 질의 실행
```bash
cd external/CloseAirCombat
python scripts/command/query_commander_rag.py \
  --db-path /tmp/commander_live.db \
  --model-id exaone4 \
  --local-exaone-path /path/to/local/exaone4 \
  --local-exaone-device cpu \
  --prompt "현재 교전 중인 지역과 아군 손실 현황을 요약해줘"
```

- 2v2 교전은 `renders/render_2v2.py` 방식과 동일하게 아군/적군 정책(`actor_<index>.pt`)을 각각 로드해 수행 가능 (`run_air_commander_system.py`의 `--ego-policy-dir`, `--enm-policy-dir`, `--ego-policy-index`, `--enm-policy-index`)

