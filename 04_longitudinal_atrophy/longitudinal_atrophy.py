"""
Longitudinal atrophy analysis.

This script calculates percentage volume change and annualized
percentage volume change between baseline and follow-up measurements
for predefined brain volumetric measures.
"""

import argparse
from datetime import datetime


def percentage_change(baseline_volume, followup_volume):
    """
    Calculate percentage volume change relative to baseline.
    """

    if baseline_volume <= 0:
        raise ValueError("Baseline volume must be greater than zero.")

    return (
        (followup_volume - baseline_volume)
        / baseline_volume
        * 100.0
    )


def annualized_percentage_change(
    baseline_volume,
    followup_volume,
    interval_years,
):
    """
    Calculate annualized percentage volume change.
    """

    if interval_years <= 0:
        raise ValueError(
            "Follow-up interval must be greater than zero."
        )

    change = percentage_change(
        baseline_volume,
        followup_volume,
    )

    return change / interval_years


def calculate_interval_years(
    baseline_date,
    followup_date,
):
    """
    Calculate follow-up interval in years from acquisition dates.

    Dates should be supplied as YYYY-MM-DD.
    """

    baseline = datetime.strptime(
        baseline_date,
        "%Y-%m-%d",
    )

    followup = datetime.strptime(
        followup_date,
        "%Y-%m-%d",
    )

    days = (followup - baseline).days

    if days <= 0:
        raise ValueError(
            "Follow-up date must be after baseline date."
        )

    return days / 365.25


def main():

    parser = argparse.ArgumentParser(
        description=(
            "Calculate longitudinal percentage and annualized "
            "volume change between baseline and follow-up MRI."
        )
    )

    parser.add_argument(
        "--baseline-volume",
        type=float,
        required=True,
        help="Baseline volume in mm^3.",
    )

    parser.add_argument(
        "--followup-volume",
        type=float,
        required=True,
        help="Follow-up volume in mm^3.",
    )

    interval_group = parser.add_mutually_exclusive_group(
        required=True
    )

    interval_group.add_argument(
        "--interval-years",
        type=float,
        help="Follow-up interval in years.",
    )

    interval_group.add_argument(
        "--baseline-date",
        type=str,
        help="Baseline acquisition date in YYYY-MM-DD format.",
    )

    parser.add_argument(
        "--followup-date",
        type=str,
        help=(
            "Follow-up acquisition date in YYYY-MM-DD format. "
            "Required when --baseline-date is used."
        ),
    )

    args = parser.parse_args()

    if args.interval_years is not None:
        interval_years = args.interval_years

    else:
        if args.followup_date is None:
            parser.error(
                "--followup-date is required when "
                "--baseline-date is provided."
            )

        interval_years = calculate_interval_years(
            args.baseline_date,
            args.followup_date,
        )

    change = percentage_change(
        args.baseline_volume,
        args.followup_volume,
    )

    annualized_change = annualized_percentage_change(
        args.baseline_volume,
        args.followup_volume,
        interval_years,
    )

    print(f"Baseline volume: {args.baseline_volume:.2f} mm³")
    print(f"Follow-up volume: {args.followup_volume:.2f} mm³")
    print(f"Follow-up interval: {interval_years:.3f} years")
    print(f"Volume change: {change:.2f}%")
    print(
        f"Annualized volume change: "
        f"{annualized_change:.2f}%/year"
    )


if __name__ == "__main__":
    main()
