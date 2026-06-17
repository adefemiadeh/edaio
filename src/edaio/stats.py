"""Parallel statistics computation engine."""

import pandas as pd
import numpy as np
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Dict, Any, Optional
import warnings

warnings.filterwarnings("ignore", category=FutureWarning)


def _compute_column_stats(col: pd.Series) -> Dict[str, Any]:
    """Compute detailed stats for a single column (parallel-ready)."""
    info = {
        "dtype": str(col.dtype),
        "null_pct": col.isna().mean(),
        "null_count": col.isna().sum(),
        "unique": col.nunique(dropna=False),
    }
    
    # --- Datetime handling ---
    if pd.api.types.is_datetime64_any_dtype(col):
        clean = col.dropna()
        if len(clean) > 0:
            info["is_datetime"] = True
            info["min"] = clean.min()
            info["max"] = clean.max()
            info["range_days"] = (clean.max() - clean.min()).days
        return info

    # --- Numeric ---
    if pd.api.types.is_numeric_dtype(col):
        clean = col.dropna()
        if len(clean) > 0:
            q1, q3 = clean.quantile(0.25), clean.quantile(0.75)
            iqr = q3 - q1
            info.update({
                "mean": clean.mean(),
                "std": clean.std(),
                "min": clean.min(),
                "max": clean.max(),
                "q25": q1,
                "q50": clean.quantile(0.50),
                "q75": q3,
                "skew": clean.skew(),
                "kurtosis": clean.kurtosis(),
                "outliers_iqr": ((clean < (q1 - 1.5 * iqr)) | (clean > (q3 + 1.5 * iqr))).sum(),
                "outlier_pct": ((clean < (q1 - 1.5 * iqr)) | (clean > (q3 + 1.5 * iqr))).mean()
            })
        return info

    # --- Categorical / Text ---
    clean = col.dropna()
    if len(clean) > 0:
        value_counts = clean.value_counts()
        info["top_values"] = value_counts.head(5).to_dict()
        info["top1_pct"] = value_counts.iloc[0] / len(clean) if len(value_counts) > 0 else 0
        # Check if it's a high-cardinality identifier
        if len(value_counts) / len(clean) > 0.95 and len(clean) > 100:
            info["likely_id"] = True
    return info


def compute_stats(df: pd.DataFrame, sample_limit: int = 100_000) -> Dict[str, Any]:
    """Parallel computation of all column stats."""
    # Downsample if huge
    if len(df) > sample_limit:
        sample_df = df.sample(sample_limit, random_state=42)
        is_sampled = True
    else:
        sample_df = df
        is_sampled = False

    stats = {"columns": {}, "is_sampled": is_sampled, "n_rows": len(df)}
    
    # Parallelize column computations
    with ThreadPoolExecutor(max_workers=8) as executor:
        future_to_col = {executor.submit(_compute_column_stats, sample_df[col]): col for col in df.columns}
        for future in as_completed(future_to_col):
            col = future_to_col[future]
            try:
                stats["columns"][col] = future.result()
            except Exception as e:
                stats["columns"][col] = {"error": str(e), "dtype": "unknown"}

    # Global stats
    stats["duplicates"] = int(df.duplicated().sum())
    stats["memory_usage_mb"] = df.memory_usage(deep=True).sum() / (1024**2)
    
    # Dtype suggestions (memory downcasting)
    stats["dtype_suggestions"] = {}
    for col, info in stats["columns"].items():
        if info.get("dtype") == "int64" and df[col].max() < 32767:
            stats["dtype_suggestions"][col] = "int16 (75% memory saving)"
        elif info.get("dtype") == "float64":
            stats["dtype_suggestions"][col] = "float32 (50% saving)"
    
    # Full correlation matrix (only if < 100 numeric cols to avoid blowup)
    num_cols = df.select_dtypes(include=np.number).columns
    if 2 <= len(num_cols) <= 100:
        stats["corr_matrix"] = df[num_cols].corr()
    
    return stats


def compute_psi(train_df: pd.DataFrame, test_df: pd.DataFrame, stats_train: Dict) -> Dict:
    """Population Stability Index for train/test comparison."""
    psi_results = {}
    for col in train_df.columns:
        if col not in test_df.columns:
            continue
        # Bin numeric into 10 equal-width bins based on train
        if pd.api.types.is_numeric_dtype(train_df[col]):
            train_clean = train_df[col].dropna()
            test_clean = test_df[col].dropna()
            if len(train_clean) < 2 or len(test_clean) < 2:
                continue
            bins = np.percentile(train_clean, np.linspace(0, 100, 11))
            bins = np.unique(bins)
            if len(bins) < 2:
                continue
            train_bin = np.digitize(train_clean, bins[:-1])
            test_bin = np.digitize(test_clean, bins[:-1])
            train_dist = np.bincount(train_bin, minlength=len(bins)) / len(train_clean)
            test_dist = np.bincount(test_bin, minlength=len(bins)) / len(test_clean)
        else:  # categorical
            cats = set(train_df[col].dropna().unique()) | set(test_df[col].dropna().unique())
            train_counts = train_df[col].value_counts()
            test_counts = test_df[col].value_counts()
            train_dist = np.array([train_counts.get(c, 0) for c in cats]) / len(train_df[col].dropna())
            test_dist = np.array([test_counts.get(c, 0) for c in cats]) / len(test_df[col].dropna())
        
        # PSI = sum((train - test) * ln(train / test))
        psi = 0
        for t, v in zip(train_dist, test_dist):
            if t > 0 and v > 0:
                psi += (t - v) * np.log(t / v)
        psi_results[col] = psi
    return psi_results