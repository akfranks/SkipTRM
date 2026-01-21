#!/usr/bin/env python3
"""
Experiment Status Checker
=========================
Checks the status of all experiments defined in experiments.yaml.

Status is determined by:
1. Checkpoint presence and progress
2. W&B run status (if available)
3. Training metadata (epoch/step)

Usage:
    python check_status.py
    python check_status.py --verbose
    python check_status.py --experiment lstm_base_h512
    python check_status.py --format json
"""

import argparse
import json
import os
from dataclasses import dataclass
from enum import Enum
from pathlib import Path
from typing import Any, Dict, List, Optional

import yaml

try:
    import wandb
    WANDB_AVAILABLE = True
except ImportError:
    WANDB_AVAILABLE = False

try:
    import torch
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False


class Status(Enum):
    PENDING = "pending"      # Not started
    RUNNING = "running"      # In progress
    COMPLETED = "completed"  # Finished all epochs
    FAILED = "failed"        # Has checkpoint but incomplete
    UNKNOWN = "unknown"      # Cannot determine


@dataclass
class ExperimentStatus:
    name: str
    status: Status
    description: str
    arch: str
    target_epochs: int
    current_epoch: int
    current_step: int
    latest_checkpoint: Optional[str]
    checkpoint_path: str
    wandb_run_id: Optional[str] = None
    message: str = ""


def load_experiments(config_path: str = "experiments.yaml") -> Dict[str, Any]:
    """Load the experiments configuration file."""
    with open(config_path, "r") as f:
        return yaml.safe_load(f)


def find_latest_checkpoint(checkpoint_path: str) -> Optional[str]:
    """Find the latest checkpoint file in the checkpoint directory."""
    if not os.path.exists(checkpoint_path):
        return None

    checkpoint_files = [f for f in os.listdir(checkpoint_path) if f.startswith("step_")]
    if not checkpoint_files:
        return None

    step_numbers = []
    for f in checkpoint_files:
        try:
            step_num = int(f.split("_")[1])
            step_numbers.append((step_num, f))
        except (ValueError, IndexError):
            continue

    if not step_numbers:
        return None

    latest = max(step_numbers, key=lambda x: x[0])
    return os.path.join(checkpoint_path, latest[1])


def load_training_metadata(checkpoint_path: str) -> Dict[str, Any]:
    """Load training metadata from checkpoint directory."""
    metadata_path = os.path.join(checkpoint_path, "training_metadata.pt")

    if not os.path.exists(metadata_path):
        return {"current_epoch": 0, "step": 0}

    if TORCH_AVAILABLE:
        try:
            metadata = torch.load(metadata_path, map_location="cpu")
            return metadata
        except Exception:
            return {"current_epoch": 0, "step": 0}

    return {"current_epoch": 0, "step": 0}


def get_experiment_status(
    name: str,
    experiment: Dict[str, Any],
    defaults: Dict[str, Any]
) -> ExperimentStatus:
    """Determine the status of a single experiment."""
    # Merge configs
    target_epochs = experiment.get("epochs", defaults.get("epochs", 50000))
    project_name = experiment.get("project_name", defaults.get("project_name", "TinyRecursiveModels"))
    arch = experiment.get("arch", "unknown")
    description = experiment.get("description", "")

    # Checkpoint path
    checkpoint_path = os.path.join("checkpoints", project_name, name)

    # Check checkpoint
    latest_checkpoint = find_latest_checkpoint(checkpoint_path)
    metadata = load_training_metadata(checkpoint_path)

    current_epoch = metadata.get("current_epoch", 0)
    current_step = metadata.get("step", 0)

    # Determine status
    if not os.path.exists(checkpoint_path):
        status = Status.PENDING
        message = "Not started"
    elif latest_checkpoint is None:
        status = Status.PENDING
        message = "Checkpoint directory exists but no checkpoints found"
    elif current_epoch >= target_epochs:
        status = Status.COMPLETED
        message = f"Completed {current_epoch}/{target_epochs} epochs"
    elif current_epoch > 0:
        progress = (current_epoch / target_epochs) * 100
        status = Status.RUNNING
        message = f"Progress: {current_epoch}/{target_epochs} epochs ({progress:.1f}%)"
    else:
        status = Status.UNKNOWN
        message = "Has checkpoints but cannot determine progress"

    return ExperimentStatus(
        name=name,
        status=status,
        description=description,
        arch=arch,
        target_epochs=target_epochs,
        current_epoch=current_epoch,
        current_step=current_step,
        latest_checkpoint=latest_checkpoint,
        checkpoint_path=checkpoint_path,
        message=message
    )


def get_all_statuses(config: Dict[str, Any]) -> List[ExperimentStatus]:
    """Get status for all experiments."""
    experiments = config.get("experiments", {})
    defaults = config.get("defaults", {})

    statuses = []
    for name, experiment in experiments.items():
        status = get_experiment_status(name, experiment, defaults)
        statuses.append(status)

    return statuses


def print_status_table(statuses: List[ExperimentStatus], verbose: bool = False) -> None:
    """Print a formatted status table."""
    # Status symbols and colors (ANSI)
    status_symbols = {
        Status.PENDING: ("○", "\033[90m"),     # Gray
        Status.RUNNING: ("◐", "\033[33m"),     # Yellow
        Status.COMPLETED: ("●", "\033[32m"),   # Green
        Status.FAILED: ("✗", "\033[31m"),      # Red
        Status.UNKNOWN: ("?", "\033[35m"),     # Magenta
    }
    RESET = "\033[0m"

    # Header
    print("\n" + "=" * 80)
    print("EXPERIMENT STATUS")
    print("=" * 80)

    # Group by status
    by_status = {}
    for s in statuses:
        by_status.setdefault(s.status, []).append(s)

    # Print summary
    print(f"\nSummary:")
    for status in [Status.COMPLETED, Status.RUNNING, Status.PENDING, Status.FAILED, Status.UNKNOWN]:
        count = len(by_status.get(status, []))
        if count > 0:
            symbol, color = status_symbols[status]
            print(f"  {color}{symbol}{RESET} {status.value:12s}: {count}")

    # Print details
    print("\n" + "-" * 80)
    print(f"{'Experiment':<30s} {'Status':<12s} {'Progress':<20s} {'Arch':<15s}")
    print("-" * 80)

    # Sort by status priority, then name
    status_order = {
        Status.RUNNING: 0,
        Status.COMPLETED: 1,
        Status.PENDING: 2,
        Status.FAILED: 3,
        Status.UNKNOWN: 4,
    }

    sorted_statuses = sorted(statuses, key=lambda s: (status_order[s.status], s.name))

    for s in sorted_statuses:
        symbol, color = status_symbols[s.status]

        if s.target_epochs > 0:
            progress = f"{s.current_epoch}/{s.target_epochs}"
        else:
            progress = "N/A"

        print(f"{s.name:<30s} {color}{symbol} {s.status.value:<10s}{RESET} {progress:<20s} {s.arch:<15s}")

        if verbose and s.message:
            print(f"    └─ {s.message}")
            if s.latest_checkpoint:
                print(f"       Checkpoint: {s.latest_checkpoint}")

    print("-" * 80)
    print(f"Total: {len(statuses)} experiments\n")


def print_json(statuses: List[ExperimentStatus]) -> None:
    """Print status as JSON."""
    data = {
        "experiments": [
            {
                "name": s.name,
                "status": s.status.value,
                "description": s.description,
                "arch": s.arch,
                "target_epochs": s.target_epochs,
                "current_epoch": s.current_epoch,
                "current_step": s.current_step,
                "checkpoint_path": s.checkpoint_path,
                "latest_checkpoint": s.latest_checkpoint,
                "message": s.message,
            }
            for s in statuses
        ],
        "summary": {
            status.value: len([s for s in statuses if s.status == status])
            for status in Status
        }
    }
    print(json.dumps(data, indent=2))


def main():
    parser = argparse.ArgumentParser(
        description="Check status of experiments",
        formatter_class=argparse.RawDescriptionHelpFormatter
    )

    parser.add_argument(
        "--config", "-c",
        type=str,
        default="experiments.yaml",
        help="Path to experiments config file"
    )
    parser.add_argument(
        "--experiment", "-e",
        type=str,
        help="Check status of specific experiment only"
    )
    parser.add_argument(
        "--verbose", "-v",
        action="store_true",
        help="Show detailed status information"
    )
    parser.add_argument(
        "--format", "-f",
        choices=["table", "json"],
        default="table",
        help="Output format"
    )
    parser.add_argument(
        "--pending-only",
        action="store_true",
        help="Only show pending experiments"
    )
    parser.add_argument(
        "--running-only",
        action="store_true",
        help="Only show running experiments"
    )

    args = parser.parse_args()

    # Load config
    if not os.path.exists(args.config):
        print(f"Error: Config file not found: {args.config}")
        return 1

    config = load_experiments(args.config)

    # Get statuses
    if args.experiment:
        experiments = config.get("experiments", {})
        defaults = config.get("defaults", {})

        if args.experiment not in experiments:
            print(f"Error: Unknown experiment '{args.experiment}'")
            return 1

        statuses = [get_experiment_status(
            args.experiment,
            experiments[args.experiment],
            defaults
        )]
    else:
        statuses = get_all_statuses(config)

    # Filter
    if args.pending_only:
        statuses = [s for s in statuses if s.status == Status.PENDING]
    elif args.running_only:
        statuses = [s for s in statuses if s.status == Status.RUNNING]

    # Output
    if args.format == "json":
        print_json(statuses)
    else:
        print_status_table(statuses, verbose=args.verbose)

    return 0


if __name__ == "__main__":
    exit(main())
