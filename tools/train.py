import argparse
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from hmrnet.engine import train  # noqa: E402
from hmrnet.utils import load_config  # noqa: E402


def main():
    parser = argparse.ArgumentParser(description="Train HMR-Net")
    parser.add_argument("--config", required=True)
    parser.add_argument("--work-dir", default=None)
    parser.add_argument("--resume", default=None)
    args = parser.parse_args()

    cfg = load_config(args.config)
    if args.work_dir:
        cfg["work_dir"] = args.work_dir
    train(cfg, resume=args.resume)


if __name__ == "__main__":
    main()
