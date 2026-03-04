#!/usr/bin/env python
import argparse
import math

from command.air_commander_system import KoreaAirCommanderSystem


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--db-path", type=str, required=True)
    parser.add_argument("--scenario-name", type=str, default="2v2/NoWeapon/HierarchySelfplay")
    parser.add_argument("--model-id", type=str, default="exaone4")
    parser.add_argument("--steps", type=int, default=300)
    parser.add_argument("--dt-seconds", type=float, default=10.0)
    return parser.parse_args()


def main():
    args = parse_args()
    system = KoreaAirCommanderSystem(
        db_path=args.db_path,
        scenario_name=args.scenario_name,
        model_id=args.model_id,
    )

    # Example: multiple enemy aircraft descending from different directions.
    system.add_enemy_wave("BANDIT_NW_01", start_km=(124.0, 39.5), heading_rad=-math.pi / 4)
    system.add_enemy_wave("BANDIT_NE_02", start_km=(130.5, 39.4), heading_rad=-3 * math.pi / 4)
    system.add_enemy_wave("BANDIT_N_03", start_km=(127.5, 39.7), heading_rad=-math.pi / 2)

    for _ in range(args.steps):
        system.step(dt_seconds=args.dt_seconds)


if __name__ == "__main__":
    main()
