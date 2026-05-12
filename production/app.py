from __future__ import annotations

import argparse
import sys
from pathlib import Path


THIS_DIR = Path(__file__).resolve().parent
SRC_DIR = THIS_DIR / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from crosscamreid.config import load_config
from crosscamreid.pipeline import run_app


def parse_args():
    parser = argparse.ArgumentParser(description="CrossCamReid production app.")
    parser.add_argument(
        "--config",
        default=str(THIS_DIR / "config" / "config.yaml"),
        help="Path to YAML config file.",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    config = load_config(args.config)
    if not config.cameras:
        sys.stderr.write(
            "ERROR: the CLI runner requires a 'cameras:' list in "
            f"{args.config}. The block is optional for the people-counting "
            "and HLS servers (which receive cameras over the wire), but "
            "app.py has no other source for them.\n"
        )
        return 2
    return run_app(config)


if __name__ == "__main__":
    raise SystemExit(main())

