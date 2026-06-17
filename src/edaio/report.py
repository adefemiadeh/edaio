"""GlanceReport class with all features."""

from dataclasses import dataclass
from typing import Optional, List, Dict, Any
import pandas as pd
import numpy as np


@dataclass
class GlanceReport:
    """Rich report object with data insights."""
    
    df: pd.DataFrame
    stats: Dict[str, Any]
    suggestions: List[str]
    target: Optional[str] = None
    compare_df: Optional[pd.DataFrame] = None
    compare_stats: Optional[Dict] = None
    use_rich: bool = True

    def __post_init__(self):
        self._narrative = None

    @property
    def null_cols(self) -> List[str]:
        """Columns with >30% missing values."""
        return [c for c, info in self.stats["columns"].items() if info.get("null_pct", 0) > 0.30]

    @property
    def high_corr_pairs(self) -> List[tuple]:
        """Feature pairs with abs(correlation) > 0.85."""
        if "corr_matrix" not in self.stats:
            return []
        corr = self.stats["corr_matrix"]
        pairs = []
        for i in range(len(corr.columns)):
            for j in range(i+1, len(corr.columns)):
                v = corr.iloc[i, j]
                if abs(v) > 0.85:
                    pairs.append((corr.columns[i], corr.columns[j], v))
        return pairs

    def narrative(self) -> str:
        """Generate plain-English summary."""
        if self._narrative:
            return self._narrative
        
        lines = []
        n = len(self.df)
        k = len(self.df.columns)
        lines.append(f"📊 Dataset Overview: {n:,} rows, {k} columns.")
        
        null_cols = self.null_cols
        if null_cols:
            lines.append(f"⚠️ {len(null_cols)} columns have >30% missing values.")
        else:
            lines.append("✅ Low missingness overall.")
        
        if self.stats.get("duplicates", 0) > 0:
            lines.append(f"🔄 {self.stats['duplicates']} duplicate rows found.")
        
        if self.target and self.target in self.df.columns:
            lines.append(f"🎯 Target '{self.target}' is numeric.")
        
        self._narrative = " ".join(lines)
        return self._narrative

    def print(self):
        """Print a readable report to console."""
        print(f"📊 edaio Report | {len(self.df):,} rows, {len(self.df.columns)} columns")
        if self.target:
            print(f"🎯 Target: {self.target}")
        print(f"🔄 Duplicates: {self.stats.get('duplicates', 0):,}")
        print(f"💾 Memory: {self.stats.get('memory_usage_mb', 0):.2f} MB")
        
        print("\n📋 Column Summary:")
        for col, info in list(self.stats["columns"].items())[:10]:
            null_pct = info.get("null_pct", 0) * 100
            dtype = info.get("dtype", "unknown")
            print(f"  {col:20} | {dtype:10} | Null {null_pct:5.1f}%")
        
        if self.suggestions:
            print("\n⚠️ Suggestions:")
            for s in self.suggestions[:5]:
                print(f"  • {s}")
        
        if len(self.df.columns) > 10:
            print(f"\n... and {len(self.df.columns) - 10} more columns")

    def to_html(self, filepath: str = "report.html"):
        """Export an HTML report (placeholder)."""
        # Simple implementation for now
        html = f"""
        <html>
        <head><title>edaio Report</title></head>
        <body>
            <h1>📊 edaio Report</h1>
            <p><strong>Rows:</strong> {len(self.df):,}</p>
            <p><strong>Columns:</strong> {len(self.df.columns)}</p>
            <p><strong>Suggestions:</strong> {len(self.suggestions)}</p>
        </body>
        </html>
        """
        with open(filepath, "w") as f:
            f.write(html)
        print(f"✅ HTML report saved to {filepath}")