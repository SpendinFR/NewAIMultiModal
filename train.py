"""Entry point used in the documentation examples."""
from __future__ import annotations

import argparse
import logging
from pathlib import Path

from newaimultimodal import load_config
from newaimultimodal.training.trainer import Trainer


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train the NewAIMultiModal reference pipeline")
    parser.add_argument("--config", type=Path, default=Path("configs/base.yaml"), help="Path to the YAML config file")
    parser.add_argument("--phase", type=str, default="phase0_pretrain", help="Name of the phase to run")
    parser.add_argument("--max_steps", type=int, default=5, help="Maximum number of iterations")
    return parser.parse_args()


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
    args = parse_args()
    config = load_config(args.config)
    trainer = Trainer(config)
    trainer.run_phase(args.phase, max_steps=args.max_steps)


if __name__ == "__main__":  # pragma: no cover
    main()
