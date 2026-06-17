"""Intelligent suggestion generation."""

from typing import List, Dict, Optional, Any
import pandas as pd
import numpy as np


def generate_suggestions(df: pd.DataFrame, stats: Dict, target: Optional[str] = None, 
                         compare_stats: Optional[Dict] = None) -> List[str]:
    """Generate actionable suggestions from the data."""
    sugg = []
    n_rows = len(df)

    # Column-level checks
    for col, info in stats["columns"].items():
        if info.get("error"):
            continue
            
        # Nulls
        null_pct = info.get("null_pct", 0)
        if null_pct > 0.30:
            sugg.append(f"🔴 {col}: {null_pct*100:.0f}% missing. Consider imputation (median/mode) or dropping.")
        elif null_pct > 0.10:
            sugg.append(f"🟡 {col}: {null_pct*100:.0f}% missing. Monitor if impactful.")
        
        # Unique ID
        if info.get("likely_id", False):
            sugg.append(f"ℹ️ {col}: ~100% unique. Likely an identifier—drop for modeling.")
        
        # Skew
        skew = info.get("skew")
        if skew and abs(skew) > 2:
            sugg.append(f"📈 {col}: Highly skewed ({skew:.2f}). Try log1p or Box-Cox transform.")
        
        # Outliers
        outlier_pct = info.get("outlier_pct", 0)
        if outlier_pct > 0.05:
            sugg.append(f"⚠️ {col}: {outlier_pct*100:.1f}% outliers detected (IQR method). Cap or winsorize.")
        
        # High cardinality categorical
        if info.get("dtype") == "object" and info.get("unique", 0) > 0.5 * n_rows and n_rows > 100:
            sugg.append(f"🟣 {col}: High cardinality ({info['unique']} unique). Use target encoding or embeddings.")

    # Duplicates
    if stats.get("duplicates", 0) > 0:
        dupes = stats["duplicates"]
        sugg.append(f"🔄 {dupes} duplicate rows ({dupes/n_rows*100:.1f}%). Drop or aggregate.")

    # Memory
    if stats.get("memory_usage_mb", 0) > 100:
        sugg.append(f"💾 Memory usage: {stats['memory_usage_mb']:.1f} MB. Downcasting suggested: {len(stats.get('dtype_suggestions', {}))} columns.")

    # Target-specific
    if target and target in stats.get("corr_matrix", pd.DataFrame()).columns:
        corr = stats["corr_matrix"][target].drop(target, errors="ignore").sort_values(ascending=False)
        if len(corr) > 0:
            top = corr.head(2)
            bottom = corr.tail(2)
            sugg.append(f"🎯 Top driver for '{target}': {top.index[0]} ({top.values[0]:.2f})")
            if abs(bottom.values[0]) > 0.3:
                sugg.append(f"🔻 Negative driver: {bottom.index[0]} ({bottom.values[0]:.2f})")

    # Drift (if comparing)
    if compare_stats:
        psi = compare_stats.get("psi", {})
        for col, psi_val in psi.items():
            if psi_val > 0.2:
                sugg.append(f"🌊 DRIFT: '{col}' changed significantly (PSI={psi_val:.3f}). Retrain or monitor.")
            elif psi_val > 0.1:
                sugg.append(f"🌊 Mild drift in '{col}' (PSI={psi_val:.3f}).")

    return sugg