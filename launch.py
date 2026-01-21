#!/usr/bin/env python3
"""
Unified Experiment Launcher
===========================
Reads experiments.yaml and launches training with the appropriate configuration.

Usage:
    python launch.py --experiment lstm_base_h512
    python launch.py --experiment lstm_base_h512 --dry-run
    python launch.py --list
    python launch.py --experiment lstm_base_h512 --override epochs=100000
"""

import argparse
import os
import subprocess
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional

import yaml


def load_experiments(config_path: str = "experiments.yaml") -> Dict[str, Any]:
    """Load the experiments configuration file."""
    with open(config_path, "r") as f:
        return yaml.safe_load(f)


def merge_configs(defaults: Dict[str, Any], experiment: Dict[str, Any]) -> Dict[str, Any]:
    """Merge experiment config with defaults."""
    config = defaults.copy()

    # Update with experiment-level settings (non-overrides)
    for key, value in experiment.items():
        if key not in ("description", "overrides", "arch"):
            config[key] = value

    return config


def build_hydra_overrides(
    arch: str,
    config: Dict[str, Any],
    overrides: Dict[str, Any],
    run_name: str,
    cli_overrides: List[str]
) -> List[str]:
    """Build the list of Hydra command-line overrides."""
    args = []

    # Architecture
    args.append(f"arch={arch}")

    # Standard config keys that map directly to Hydra
    direct_keys = [
        "data_paths", "epochs", "eval_interval", "global_batch_size",
        "lr", "puzzle_emb_lr", "weight_decay", "puzzle_emb_weight_decay",
        "ema", "checkpoint_every_eval", "ema_rate", "seed",
        "min_eval_interval", "freeze_weights", "data_paths_test"
    ]

    for key in direct_keys:
        if key in config:
            value = config[key]
            # Handle lists specially for Hydra
            if isinstance(value, list):
                # Format as Hydra list syntax: [item1,item2,...]
                list_str = "[" + ",".join(str(v) for v in value) + "]"
                args.append(f"{key}={list_str}")
            elif isinstance(value, bool):
                args.append(f"{key}={str(value)}")
            else:
                args.append(f"{key}={value}")

    # Project and run name (use + prefix for new keys)
    if "project_name" in config:
        args.append(f'+project_name="{config["project_name"]}"')
    args.append(f'+run_name="{run_name}"')

    # Architecture-specific overrides (use arch. prefix)
    for key, value in overrides.items():
        if isinstance(value, list):
            # Convert list to Hydra format
            args.append(f"arch.{key}={value}")
        elif isinstance(value, bool):
            args.append(f"arch.{key}={str(value)}")
        else:
            args.append(f"arch.{key}={value}")

    # CLI overrides (highest priority)
    args.extend(cli_overrides)

    return args


def get_checkpoint_path(project_name: str, run_name: str) -> str:
    """Get the checkpoint path for an experiment."""
    return os.path.join("checkpoints", project_name, run_name)


def list_experiments(config: Dict[str, Any]) -> None:
    """Print all available experiments."""
    print("\nAvailable experiments:")
    print("=" * 60)

    experiments = config.get("experiments", {})

    # Group by architecture type
    grouped = {}
    for name, exp in experiments.items():
        arch = exp.get("arch", "unknown")
        arch_base = arch.split("_")[0] if "_" in arch else arch
        grouped.setdefault(arch_base, []).append((name, exp))

    for arch_type in sorted(grouped.keys()):
        print(f"\n{arch_type.upper()}:")
        for name, exp in sorted(grouped[arch_type]):
            desc = exp.get("description", "No description")
            print(f"  {name:30s} - {desc}")

    print("\n" + "=" * 60)
    print(f"Total: {len(experiments)} experiments")


def main():
    parser = argparse.ArgumentParser(
        description="Unified experiment launcher for TinyRecursiveModels",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    python launch.py --experiment lstm_base_h512
    python launch.py --experiment lstm_base_h512 --dry-run
    python launch.py --list
    python launch.py --experiment lstm_base_h512 --override epochs=100000
        """
    )

    parser.add_argument(
        "--experiment", "-e",
        type=str,
        help="Name of the experiment to run"
    )
    parser.add_argument(
        "--list", "-l",
        action="store_true",
        help="List all available experiments"
    )
    parser.add_argument(
        "--dry-run", "-n",
        action="store_true",
        help="Print the command without executing"
    )
    parser.add_argument(
        "--config", "-c",
        type=str,
        default="experiments.yaml",
        help="Path to experiments config file"
    )
    parser.add_argument(
        "--override", "-o",
        action="append",
        default=[],
        help="Additional Hydra overrides (can be used multiple times)"
    )
    parser.add_argument(
        "--run-name-suffix",
        type=str,
        default="",
        help="Suffix to add to the run name (useful for multiple runs)"
    )

    args = parser.parse_args()

    # Load config
    if not os.path.exists(args.config):
        print(f"Error: Config file not found: {args.config}")
        sys.exit(1)

    config = load_experiments(args.config)

    # List mode
    if args.list:
        list_experiments(config)
        return

    # Require experiment name
    if not args.experiment:
        parser.print_help()
        print("\nError: --experiment is required (or use --list to see options)")
        sys.exit(1)

    # Get experiment config
    experiments = config.get("experiments", {})
    if args.experiment not in experiments:
        print(f"Error: Unknown experiment '{args.experiment}'")
        print("Use --list to see available experiments")
        sys.exit(1)

    experiment = experiments[args.experiment]
    defaults = config.get("defaults", {})

    # Merge configs
    merged_config = merge_configs(defaults, experiment)

    # Build run name
    run_name = args.experiment
    if args.run_name_suffix:
        run_name = f"{run_name}_{args.run_name_suffix}"

    # Build command
    arch = experiment.get("arch", "trm")
    overrides = experiment.get("overrides", {})

    hydra_args = build_hydra_overrides(
        arch=arch,
        config=merged_config,
        overrides=overrides,
        run_name=run_name,
        cli_overrides=args.override
    )

    cmd = ["python3", "pretrain.py"] + hydra_args

    # Print or execute
    if args.dry_run:
        print("\nDry run - would execute:")
        print("=" * 60)
        print("python3 pretrain.py \\")
        for i, arg in enumerate(hydra_args):
            suffix = " \\" if i < len(hydra_args) - 1 else ""
            print(f"    {arg}{suffix}")
        print("=" * 60)

        # Show checkpoint path
        project_name = merged_config.get("project_name", "TinyRecursiveModels")
        checkpoint_path = get_checkpoint_path(project_name, run_name)
        print(f"\nCheckpoint path: {checkpoint_path}")
    else:
        print(f"\nLaunching experiment: {args.experiment}")
        print(f"Run name: {run_name}")
        print("-" * 40)

        # Execute
        try:
            subprocess.run(cmd, check=True)
        except subprocess.CalledProcessError as e:
            print(f"\nExperiment failed with exit code {e.returncode}")
            sys.exit(e.returncode)
        except KeyboardInterrupt:
            print("\nExperiment interrupted by user")
            sys.exit(130)


if __name__ == "__main__":
    main()
