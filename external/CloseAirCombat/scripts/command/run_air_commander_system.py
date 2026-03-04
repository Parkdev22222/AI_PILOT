#!/usr/bin/env python
import argparse
import os
import sys
import math

sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.realpath(__file__)))))

from command.air_commander_system import KoreaAirCommanderSystem


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--db-path", type=str, required=True)
    parser.add_argument("--scenario-name", type=str, default="2v2/NoWeapon/HierarchySelfplay")
    parser.add_argument("--model-id", type=str, default="exaone4")
    parser.add_argument("--local-exaone-path", type=str, default="")
    parser.add_argument("--local-exaone-device", type=str, default="cpu")
    parser.add_argument("--steps", type=int, default=300)
    parser.add_argument("--dt-seconds", type=float, default=10.0)
    parser.add_argument("--ego-policy-dir", type=str, default="")
    parser.add_argument("--enm-policy-dir", type=str, default="")
    parser.add_argument("--ego-policy-index", type=str, default="latest")
    parser.add_argument("--enm-policy-index", type=str, default="latest")
    parser.add_argument("--policy-device", type=str, default="cpu")
    return parser.parse_args()


def main():
    args = parse_args()
    system = KoreaAirCommanderSystem(
        db_path=args.db_path,
        scenario_name=args.scenario_name,
        model_id=args.model_id,
        local_exaone_path=args.local_exaone_path,
        local_exaone_device=args.local_exaone_device,
        ego_policy_dir=args.ego_policy_dir,
        enm_policy_dir=args.enm_policy_dir,
        ego_policy_index=args.ego_policy_index,
        enm_policy_index=args.enm_policy_index,
        policy_device=args.policy_device,
    )

    # Example: multiple enemy aircraft descending from different directions.
    system.add_enemy_wave("BANDIT_NW_01", start_lon_lat=(124.0, 39.5), heading_rad=-math.pi / 4)
    system.add_enemy_wave("BANDIT_NE_02", start_lon_lat=(130.5, 39.4), heading_rad=-3 * math.pi / 4)
    system.add_enemy_wave("BANDIT_N_03", start_lon_lat=(127.5, 39.7), heading_rad=-math.pi / 2)

    print(f"run_id={system.run_id}")
    for _ in range(args.steps):
        system.step(dt_seconds=args.dt_seconds)


if __name__ == "__main__":
    main()
