"""Pytest configuration and shared fixtures."""

import pytest
import pandas as pd
import numpy as np


@pytest.fixture
def sample_df():
    """Create a sample DataFrame for testing."""
    np.random.seed(42)
    return pd.DataFrame({
        'age': np.random.normal(35, 10, 100).astype(int),
        'income': np.random.exponential(50000, 100),
        'category': np.random.choice(['A', 'B', 'C'], 100),
        'target': np.random.normal(100, 20, 100)
    })


@pytest.fixture
def clean_df():
    """Return a clean DataFrame with no issues."""
    return pd.DataFrame({
        'x': [1, 2, 3, 4, 5],
        'y': [10, 20, 30, 40, 50],
        'z': ['a', 'b', 'c', 'd', 'e']
    })


@pytest.fixture
def messy_df():
    """Return a DataFrame with common data quality issues."""
    np.random.seed(123)
    df = pd.DataFrame({
        'id': range(100),
        'age': np.random.normal(30, 10, 100).astype(int),
        'salary': np.random.exponential(50000, 100),
        'department': np.random.choice(['HR', 'IT', 'Sales', 'Marketing', np.nan], 100),
        'target': np.random.normal(100, 20, 100)
    })
    # Inject issues
    df.loc[10:20, 'age'] = np.nan
    df.loc[30:35, 'salary'] = np.nan
    df.loc[5:8, 'department'] = np.nan
    df.loc[40:45, 'salary'] = 9999999  # Outliers
    # Add duplicates
    df = pd.concat([df, df.iloc[:5]])
    return df.reset_index(drop=True)


@pytest.fixture
def large_df():
    """Create a large DataFrame for performance testing."""
    np.random.seed(42)
    n = 10000
    return pd.DataFrame({
        'col1': np.random.normal(0, 1, n),
        'col2': np.random.exponential(1, n),
        'col3': np.random.choice(['A', 'B', 'C', 'D'], n),
        'col4': np.random.randint(0, 100, n),
        'col5': np.random.randn(n),
    })


@pytest.fixture
def sample_df_compare():
    """Create a comparison dataset with drift."""
    np.random.seed(123)
    n = 100
    return pd.DataFrame({
        'age': np.random.normal(45, 12, n),  # shifted distribution
        'income': np.random.exponential(40000, n),
        'category': np.random.choice(['A', 'B', 'X', 'Y'], n, p=[0.1, 0.1, 0.4, 0.4]),
        'target': np.random.normal(110, 25, n)
    })