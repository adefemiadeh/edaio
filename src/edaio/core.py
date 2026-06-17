"""Core edaio functionality."""

from typing import Optional, List, Any
import pandas as pd
from .stats import compute_stats, compute_psi
from .suggestions import generate_suggestions
from .report import GlanceReport


def glance(
    df: pd.DataFrame,
    target: Optional[str] = None,
    compare_to: Optional[pd.DataFrame] = None,
    columns: Optional[List[str]] = None,
    sample_limit: int = 100_000,
    use_rich: bool = True,
) -> GlanceReport:
    """
    Quick DataFrame overview with smart suggestions.
    
    Parameters
    ----------
    df : pd.DataFrame
        DataFrame to analyze
    target : str, optional
        Target column for supervised insights
    compare_to : pd.DataFrame, optional
        Second dataset for drift detection
    columns : list, optional
        Subset of columns to analyze
    sample_limit : int
        Rows to sample for heavy quantile stats
    use_rich : bool
        Enable rich console output
    
    Returns
    -------
    GlanceReport
        Rich report object with stats, suggestions, and insights
    """
    # Subset columns safely
    if columns:
        if len(columns) == 1:
            df = df[[columns[0]]]  # Keep as DataFrame
            if compare_to is not None:
                compare_to = compare_to[[columns[0]]]
        else:
            df = df[columns]
            if compare_to is not None:
                compare_to = compare_to[columns]

    # Compute stats (parallelized)
    stats = compute_stats(df, sample_limit)
    compare_stats = None
    if compare_to is not None:
        compare_stats = compute_stats(compare_to, sample_limit)
        psi = compute_psi(df, compare_to, stats)
        compare_stats["psi"] = psi

    # Generate suggestions
    suggestions = generate_suggestions(df, stats, target, compare_stats)

    # Build report
    report = GlanceReport(
        df=df,
        stats=stats,
        suggestions=suggestions,
        target=target,
        compare_df=compare_to,
        compare_stats=compare_stats,
        use_rich=use_rich
    )
    report.print()
    return report