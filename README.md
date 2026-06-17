# edaio ⚡

**Lightning-fast DataFrame introspection with smart red-flag suggestions.**

[![PyPI version](https://badge.fury.io/py/edaio.svg)](https://badge.fury.io/py/edaio)
[![Python versions](https://img.shields.io/pypi/pyversions/edaio.svg)](https://pypi.org/project/edaio/)
[![Tests](https://github.com/adefemiadeh/edaio/actions/workflows/publish.yml/badge.svg)](https://github.com/adefemiadeh/edaio/actions)
[![License](https://img.shields.io/badge/license-MIT-blue.svg)](https://opensource.org/licenses/MIT)

## 🚀 Quick Start

```python
import pandas as pd
from edaio import glance

df = pd.read_csv("your_data.csv")
report = glance(df, target='price')
```

✨ Features

- Instant insights – One line of code, zero configuration

- Actionable suggestions – Tells you exactly what to fix

- Parallel computation – Lightning fast on large datasets

- Drift detection – Compare train/test datasets with PSI

- Programmable API – Query the report object for specific issues

- Lightweight – No heavy dependencies or bloated reports

📦 Installation

```
pip install edaio
```

With optional features:

```
pip install edaio[rich,plotly]
```

📊 Examples
Basic Usage

```
from edaio import glance
import pandas as pd

df = pd.read_csv("sales.csv")
report = glance(df)
```

Target Analysis

```
report = glance(df, target='price')
print(report.null_cols)  # Columns with >30% missing
print(report.high_corr_pairs)  # Highly correlated features
```

Train/Test Comparison

```
report = glance(train_df, compare_to=test_df)
```

See drift detection in the suggestions

🔧 Command Line

```
edaio data.csv --target price --output-html report.html
```
