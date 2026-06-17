"""Command-line interface for edaio."""

import argparse
import sys
from pathlib import Path
import pandas as pd
from .core import glance


def main():
    """Main entry point for the CLI."""
    parser = argparse.ArgumentParser(
        description="⚡ edaio - Lightning-fast DataFrame introspection.",
        usage="edaio <file> [--target TARGET] [--compare FILE] [--columns COLS]"
    )
    parser.add_argument(
        "file",
        help="Path to CSV, Parquet, or Excel file."
    )
    parser.add_argument(
        "--target", "-t",
        help="Target column for supervised insights."
    )
    parser.add_argument(
        "--compare", "-c",
        help="Path to comparison dataset (train/test drift detection)."
    )
    parser.add_argument(
        "--columns",
        help="Comma-separated columns to analyze (e.g., 'age,income,gender')."
    )
    parser.add_argument(
        "--no-rich",
        action="store_true",
        help="Disable rich console output (fallback to plain text)."
    )
    parser.add_argument(
        "--output-html", "-o",
        help="Export report to HTML file."
    )
    parser.add_argument(
        "--output-clean",
        help="Export auto-clean script to .py file."
    )
    parser.add_argument(
        "--sample-limit",
        type=int,
        default=100_000,
        help="Rows to sample for heavy quantile stats (default: 100k)."
    )
    args = parser.parse_args()

    # --- Load main dataset ---
    file_path = Path(args.file)
    if not file_path.exists():
        print(f"❌ File not found: {args.file}")
        sys.exit(1)

    ext = file_path.suffix.lower()
    if ext == ".csv":
        df = pd.read_csv(file_path)
    elif ext == ".parquet":
        df = pd.read_parquet(file_path)
    elif ext in [".xlsx", ".xls"]:
        df = pd.read_excel(file_path)
    else:
        print(f"❌ Unsupported file type: {ext}. Use CSV, Parquet, or Excel.")
        sys.exit(1)

    print(f"✅ Loaded {len(df):,} rows, {len(df.columns)} columns from {file_path.name}")

    # --- Load comparison dataset ---
    compare_df = None
    if args.compare:
        compare_path = Path(args.compare)
        if not compare_path.exists():
            print(f"❌ Comparison file not found: {args.compare}")
            sys.exit(1)
        ext_cmp = compare_path.suffix.lower()
        if ext_cmp == ".csv":
            compare_df = pd.read_csv(compare_path)
        elif ext_cmp == ".parquet":
            compare_df = pd.read_parquet(compare_path)
        elif ext_cmp in [".xlsx", ".xls"]:
            compare_df = pd.read_excel(compare_path)
        else:
            print(f"❌ Unsupported file type for comparison: {ext_cmp}")
            sys.exit(1)
        print(f"✅ Loaded comparison: {len(compare_df):,} rows")

    # --- Parse columns ---
    cols = None
    if args.columns:
        cols = [c.strip() for c in args.columns.split(",")]

    # --- Run glance ---
    report = glance(
        df=df,
        target=args.target,
        compare_to=compare_df,
        columns=cols,
        sample_limit=args.sample_limit,
        use_rich=not args.no_rich
    )

    # --- Export options ---
    if args.output_html:
        report.to_html(args.output_html)

    if args.output_clean:
        report.export_clean_script(args.output_clean) # type: ignore


if __name__ == "__main__":
    main()