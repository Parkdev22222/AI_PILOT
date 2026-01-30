# Directory structure (including external repos)

```
AI_PILOT/
  external/
    CloseAirCombat/        # from https://github.com/snu-larr/CloseAirCombat.git
    CLAM-RL/               # from https://github.com/WenhaoMa-UTS/CLAM-RL.git
  configs/
    clam_closeaircombat.yaml
  scripts/
    train_clam_2v2.py
  src/
    clam_closeaircombat/
      __init__.py
      clam_adapter.py
      config.py
      env_adapter.py
      runner.py
      utils.py
  README.md
  pyproject.toml
  .gitignore
```
