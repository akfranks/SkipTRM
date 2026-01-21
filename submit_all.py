#!/usr/bin/env python3
"""
Batch Experiment Submitter
==========================
Submit multiple experiments to SLURM based on their status.

Usage:
    python submit_all.py --pending          # Submit all pending experiments
    python submit_all.py --filter "lstm*"   # Submit experiments matching pattern
    python submit_all.py --dry-run          # Show what would be submitted
    python submit_all.py --max-jobs 5       # Limit concurrent submissions
"""

import argparse
import fnmatch
import subprocess
import sys
import time
from typing import List

from check_status import (
    ExperimentStatus,
    Status,
    get_all_statuses,
    load_experiments,
)


def submit_experiment(
    experiment_name: str,
    dry_run: bool = False,
    slurm_args: List[str] = None
) -> bool:
    """Submit a single experiment to SLURM."""
    slurm_args = slurm_args or []

    cmd = ["sbatch"] + slurm_args + ["slurm_run.sh", experiment_name]

    if dry_run:
        print(f"  [DRY RUN] Would submit: {' '.join(cmd)}")
        return True

    print(f"  Submitting: {experiment_name}")
    try:
        result = subprocess.run(cmd, capture_output=True, text=True, check=True)
        print(f"    -> {result.stdout.strip()}")
        return True
    except subprocess.CalledProcessError as e:
        print(f"    -> Failed: {e.stderr.strip()}")
        return False


def main():
    parser = argparse.ArgumentParser(
        description="Batch submit experiments to SLURM",
        formatter_class=argparse.RawDescriptionHelpFormatter
    )

    # Selection options
    parser.add_argument(
        "--pending",
        action="store_true",
        help="Submit all pending experiments"
    )
    parser.add_argument(
        "--filter", "-f",
        type=str,
        help="Filter experiments by glob pattern (e.g., 'lstm*', '*h512*')"
    )
    parser.add_argument(
        "--experiments", "-e",
        nargs="+",
        help="Specific experiments to submit"
    )

    # Submission options
    parser.add_argument(
        "--dry-run", "-n",
        action="store_true",
        help="Show what would be submitted without actually submitting"
    )
    parser.add_argument(
        "--max-jobs", "-m",
        type=int,
        default=0,
        help="Maximum number of jobs to submit (0 = unlimited)"
    )
    parser.add_argument(
        "--delay",
        type=float,
        default=1.0,
        help="Delay between submissions in seconds"
    )

    # SLURM options
    parser.add_argument(
        "--partition", "-p",
        type=str,
        help="SLURM partition to use"
    )
    parser.add_argument(
        "--time", "-t",
        type=str,
        help="SLURM time limit (e.g., '24:00:00')"
    )

    # Config
    parser.add_argument(
        "--config", "-c",
        type=str,
        default="experiments.yaml",
        help="Path to experiments config file"
    )

    args = parser.parse_args()

    # Load experiments
    config = load_experiments(args.config)
    all_statuses = get_all_statuses(config)

    # Filter experiments
    to_submit: List[ExperimentStatus] = []

    if args.experiments:
        # Specific experiments
        names = set(args.experiments)
        to_submit = [s for s in all_statuses if s.name in names]
    elif args.pending:
        # All pending
        to_submit = [s for s in all_statuses if s.status == Status.PENDING]
    elif args.filter:
        # Pattern match
        to_submit = [s for s in all_statuses if fnmatch.fnmatch(s.name, args.filter)]
    else:
        parser.print_help()
        print("\nError: Specify --pending, --filter, or --experiments")
        return 1

    if not to_submit:
        print("No experiments to submit")
        return 0

    # Apply max jobs limit
    if args.max_jobs > 0:
        to_submit = to_submit[:args.max_jobs]

    # Build SLURM args
    slurm_args = []
    if args.partition:
        slurm_args.extend(["--partition", args.partition])
    if args.time:
        slurm_args.extend(["--time", args.time])

    # Submit
    print(f"\nSubmitting {len(to_submit)} experiment(s):")
    print("=" * 50)

    successful = 0
    failed = 0

    for i, status in enumerate(to_submit):
        if i > 0 and not args.dry_run:
            time.sleep(args.delay)

        success = submit_experiment(
            status.name,
            dry_run=args.dry_run,
            slurm_args=slurm_args
        )

        if success:
            successful += 1
        else:
            failed += 1

    # Summary
    print("=" * 50)
    if args.dry_run:
        print(f"Would submit {successful} experiment(s)")
    else:
        print(f"Submitted: {successful}, Failed: {failed}")

    return 0 if failed == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
