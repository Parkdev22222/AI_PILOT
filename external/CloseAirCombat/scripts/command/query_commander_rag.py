#!/usr/bin/env python
import argparse
import os
import sys

sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.realpath(__file__)))))

from command.air_commander_system import Exaone4CommanderAgent
from command.commander_db import CommanderCombatDB


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--db-path", type=str, required=True)
    parser.add_argument("--prompt", type=str, required=True)
    parser.add_argument("--model-id", type=str, default="exaone4")
    return parser.parse_args()


def main():
    args = parse_args()
    db = CommanderCombatDB(args.db_path)
    agent = Exaone4CommanderAgent(db, model_id=args.model_id)
    print(agent.decide_scramble(args.prompt))


if __name__ == "__main__":
    main()
