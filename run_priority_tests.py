#!/usr/bin/env python3
"""
Priority Test Runner
====================
Runs experiments in priority order from experiments.yaml.

Features:
- Runs experiments by priority group or priority level
- Can run locally or submit to SLURM
- Skips already completed/running experiments
- Supports dry-run mode

Usage:
    # List all priority groups
    python run_priority_tests.py --list-groups

    # Run all critical experiments (priority 1)
    python run_priority_tests.py --priority 1

    # Run a specific group
    python run_priority_tests.py --group critical

    # Run all pending experiments in priority order
    python run_priority_tests.py --all-pending

    # Submit to SLURM
    python run_priority_tests.py --priority 1 --slurm

    # Dry run
    python run_priority_tests.py --priority 1 --dry-run
"""

import argparse
import os
import subprocess
import sys
from typing import Any, Dict, List, Optional, Tuple

import yaml


def load_experiments(config_path: str = "experiments.yaml") -> Dict[str, Any]:
    """Load the experiments configuration file."""
    with open(config_path, "r") as f:
        return yaml.safe_load(f)


def get_experiment_status(name: str, project_name: str) -> str:
    """Check if experiment has checkpoints (simple status check)."""
    checkpoint_path = os.path.join("checkpoints", project_name, name)

    if not os.path.exists(checkpoint_path):
        return "pending"

    # Check for step checkpoints
    checkpoint_files = [f for f in os.listdir(checkpoint_path) if f.startswith("step_")]
    if not checkpoint_files:
        return "pending"

    # Check training metadata for completion
    metadata_path = os.path.join(checkpoint_path, "training_metadata.pt")
    if os.path.exists(metadata_path):
        try:
            import torch
            metadata = torch.load(metadata_path, map_location="cpu")
            current_epoch = metadata.get("current_epoch", 0)
            # We'd need target epochs from config, but this is a rough check
            if current_epoch > 0:
                return "running"
        except Exception:
            pass

    return "running"


def get_experiments_by_priority(
    experiments: Dict[str, Any],
    priority: int,
    defaults: Dict[str, Any]
) -> List[Tuple[str, Dict[str, Any]]]:
    """Get experiments filtered by priority level."""
    result = []
    for name, exp in experiments.items():
        if exp.get("priority", 4) == priority:
            result.append((name, exp))
    return sorted(result, key=lambda x: x[0])


def get_experiments_by_group(
    experiments: Dict[str, Any],
    priority_groups: Dict[str, Any],
    group_name: str
) -> List[Tuple[str, Dict[str, Any]]]:
    """Get experiments from a specific priority group."""
    if group_name not in priority_groups:
        return []

    group = priority_groups[group_name]
    exp_names = group.get("experiments", [])

    result = []
    for name in exp_names:
        if name in experiments:
            result.append((name, experiments[name]))
    return result


def list_priority_groups(priority_groups: Dict[str, Any]) -> None:
    """Print available priority groups."""
    print("\nAvailable priority groups:")
    print("=" * 60)
    for name, group in priority_groups.items():
        desc = group.get("description", "No description")
        exp_count = len(group.get("experiments", []))
        print(f"  {name:20s} ({exp_count} experiments)")
        print(f"      {desc}")
    print("=" * 60)


def list_experiments_by_priority(experiments: Dict[str, Any]) -> None:
    """Print experiments grouped by priority."""
    print("\nExperiments by priority:")
    print("=" * 60)

    by_priority = {}
    for name, exp in experiments.items():
        priority = exp.get("priority", 4)
        by_priority.setdefault(priority, []).append((name, exp))

    for priority in sorted(by_priority.keys()):
        exps = by_priority[priority]
        print(f"\nPriority {priority} ({len(exps)} experiments):")
        for name, exp in sorted(exps, key=lambda x: x[0]):
            desc = exp.get("description", "")[:50]
            category = exp.get("category", "unknown")
            print(f"  {name:35s} [{category}]")

    print("\n" + "=" * 60)


def run_experiment(
    name: str,
    slurm: bool = False,
    dry_run: bool = False
) -> bool:
    """Run a single experiment."""
    if slurm:
        cmd = ["sbatch", "slurm_run.sh", name]
    else:
        cmd = ["python3", "launch.py", "--experiment", name]

    if dry_run:
        print(f"  [DRY RUN] Would execute: {' '.join(cmd)}")
        return True

    print(f"  Executing: {' '.join(cmd)}")
    try:
        result = subprocess.run(cmd, check=True)
        return True
    except subprocess.CalledProcessError as e:
        print(f"  ERROR: Command failed with exit code {e.returncode}")
        return False
    except FileNotFoundError as e:
        print(f"  ERROR: Command not found: {e}")
        return False


def run_experiments(
    experiments: List[Tuple[str, Dict[str, Any]]],
    defaults: Dict[str, Any],
    slurm: bool = False,
    dry_run: bool = False,
    skip_running: bool = True
) -> Dict[str, str]:
    """Run a list of experiments."""
    project_name = defaults.get("project_name", "TinyRecursiveModels")

    results = {}
    for name, exp in experiments:
        print(f"\n--- {name} ---")
        print(f"    Description: {exp.get('description', 'N/A')}")
        print(f"    Architecture: {exp.get('arch', 'N/A')}")
        print(f"    Category: {exp.get('category', 'N/A')}")

        # Check status
        if skip_running:
            status = get_experiment_status(name, project_name)
            if status != "pending":
                print(f"    Status: {status} (skipping)")
                results[name] = f"skipped ({status})"
                continue

        # Run experiment
        success = run_experiment(name, slurm=slurm, dry_run=dry_run)
        results[name] = "launched" if success else "failed"

    return results


def main():
    parser = argparse.ArgumentParser(
        description="Run experiments by priority",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    python run_priority_tests.py --list-groups
    python run_priority_tests.py --list-by-priority
    python run_priority_tests.py --priority 1
    python run_priority_tests.py --group critical
    python run_priority_tests.py --all-pending
    python run_priority_tests.py --priority 1 --slurm
    python run_priority_tests.py --priority 1 --dry-run
        """
    )

    parser.add_argument(
        "--config", "-c",
        type=str,
        default="experiments.yaml",
        help="Path to experiments config file"
    )
    parser.add_argument(
        "--list-groups",
        action="store_true",
        help="List available priority groups"
    )
    parser.add_argument(
        "--list-by-priority",
        action="store_true",
        help="List experiments grouped by priority"
    )
    parser.add_argument(
        "--priority", "-p",
        type=int,
        help="Run experiments with this priority level (1=critical, 4=low)"
    )
    parser.add_argument(
        "--group", "-g",
        type=str,
        help="Run experiments from this priority group"
    )
    parser.add_argument(
        "--all-pending",
        action="store_true",
        help="Run all pending experiments in priority order"
    )
    parser.add_argument(
        "--slurm",
        action="store_true",
        help="Submit to SLURM instead of running locally"
    )
    parser.add_argument(
        "--dry-run", "-n",
        action="store_true",
        help="Print commands without executing"
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Run even if experiment is already running/completed"
    )

    args = parser.parse_args()

    # Load config
    if not os.path.exists(args.config):
        print(f"Error: Config file not found: {args.config}")
        sys.exit(1)

    config = load_experiments(args.config)
    experiments = config.get("experiments", {})
    defaults = config.get("defaults", {})
    priority_groups = config.get("priority_groups", {})

    # List modes
    if args.list_groups:
        list_priority_groups(priority_groups)
        return 0

    if args.list_by_priority:
        list_experiments_by_priority(experiments)
        return 0

    # Determine which experiments to run
    to_run = []

    if args.priority is not None:
        print(f"\nRunning priority {args.priority} experiments...")
        to_run = get_experiments_by_priority(experiments, args.priority, defaults)
    elif args.group:
        print(f"\nRunning experiments from group '{args.group}'...")
        to_run = get_experiments_by_group(experiments, priority_groups, args.group)
        if not to_run:
            print(f"Error: Unknown group '{args.group}'")
            list_priority_groups(priority_groups)
            return 1
    elif args.all_pending:
        print("\nRunning all pending experiments in priority order...")
        # Get all experiments sorted by priority
        all_exps = [(name, exp) for name, exp in experiments.items()]
        to_run = sorted(all_exps, key=lambda x: (x[1].get("priority", 4), x[0]))
    else:
        parser.print_help()
        print("\nError: Specify --priority, --group, or --all-pending")
        return 1

    if not to_run:
        print("No experiments to run.")
        return 0

    print(f"\nFound {len(to_run)} experiments to run")
    print("=" * 60)

    # Run experiments
    results = run_experiments(
        to_run,
        defaults,
        slurm=args.slurm,
        dry_run=args.dry_run,
        skip_running=not args.force
    )

    # Summary
    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)
    launched = sum(1 for v in results.values() if v == "launched")
    skipped = sum(1 for v in results.values() if v.startswith("skipped"))
    failed = sum(1 for v in results.values() if v == "failed")

    print(f"  Launched: {launched}")
    print(f"  Skipped:  {skipped}")
    print(f"  Failed:   {failed}")

    return 0 if failed == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
