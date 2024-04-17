"""Convert raw episodic returns to D4RL normalized scores.

Usage:
    python eval_score.py Hopper 1179.08
    python eval_score.py HalfCheetah 4763.2 Walker2d 3663.12
"""
import argparse

from utils import get_d4rl_normalized_score, REF_MAX_SCORE


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("pairs", nargs="+",
                        help="alternating <env> <return> pairs, e.g. Hopper 1179.08")
    args = parser.parse_args()

    if len(args.pairs) % 2 != 0:
        parser.error("arguments must come in <env> <return> pairs")

    for env_name, raw in zip(args.pairs[::2], args.pairs[1::2]):
        if env_name not in REF_MAX_SCORE:
            parser.error(f"unknown env '{env_name}', expected one of {sorted(REF_MAX_SCORE)}")
        score = get_d4rl_normalized_score(env_name, float(raw))
        print(f"{env_name}: raw return {float(raw):.2f} -> D4RL score {score:.2f}")


if __name__ == "__main__":
    main()
