# CLAM 2v2 Close Air Combat Integration

이 저장소는 **CloseAirCombat 2vs2 공중교전 환경**에서 **CLAM 알고리즘**을 학습하기 위한 통합 코드 스캐폴딩입니다.
외부 저장소(`CloseAirCombat`, `CLAM-RL`)를 로컬에 두고 동작하도록 설계되어 있으며,
환경 생성 함수와 CLAM 알고리즘 클래스를 유연하게 연결할 수 있습니다.

> ⚠️ 현재 환경에서는 GitHub 접근이 차단되어 두 저장소를 자동으로 가져올 수 없습니다.
> 아래 안내에 따라 로컬에서 직접 클론해 주세요.

## 1) 외부 저장소 준비

```
external/
  CloseAirCombat/   # https://github.com/snu-larr/CloseAirCombat
  CLAM-RL/          # https://github.com/WenhaoMa-UTS/CLAM-RL
```

예시:

```bash
git clone https://github.com/snu-larr/CloseAirCombat external/CloseAirCombat

git clone https://github.com/WenhaoMa-UTS/CLAM-RL external/CLAM-RL
```

## 2) 실행

환경 생성 함수 및 CLAM 알고리즘 클래스를 문자열로 지정합니다.
(예: `module:function` 또는 `module:ClassName`)

```bash
python scripts/train_clam_aircombat.py \
  --closeaircombat-path external/CloseAirCombat \
  --clam-rl-path external/CLAM-RL \
  --env-factory closeaircombat.envs.aircombat_2v2:make_env \
  --env-config configs/clam_2v2.yaml \
  --clam-class clam.algorithms.clam:CLAM \
  --output-dir runs/clam_2v2
```

- `--env-factory`와 `--clam-class`는 실제 저장소 코드에 맞게 수정해야 합니다.
- `configs/clam_2v2.yaml`에는 환경 및 학습 하이퍼파라미터가 들어갑니다.

## 3) 구성 파일 예시
`configs/clam_2v2.yaml`을 참고하세요.
