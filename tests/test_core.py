"""Complete tests for edaio."""

import pytest
import pandas as pd
from edaio import glance, GlanceReport # type: ignore


def test_glance_works(clean_df):
    """Test basic glance functionality."""
    report = glance(clean_df)
    assert isinstance(report, GlanceReport)
    assert report.df is not None


def test_glance_with_target(sample_df):
    """Test target parameter."""
    report = glance(sample_df, target='target')
    assert report.target == 'target'


def test_glance_with_columns_subset(sample_df):
    """Test columns parameter."""
    report = glance(sample_df, columns=['age', 'income'])
    assert set(report.df.columns) == {'age', 'income'}


def test_glance_suggestions(messy_df):
    """Test that suggestions are generated."""
    report = glance(messy_df)
    assert len(report.suggestions) >= 1


def test_report_properties(sample_df):
    """Test GlanceReport properties."""
    report = glance(sample_df)
    assert report.narrative() is not None
    assert isinstance(report.null_cols, list)
    assert isinstance(report.high_corr_pairs, list)


def test_glance_with_compare(sample_df, sample_df_compare):
    """Test comparison mode."""
    report = glance(sample_df, compare_to=sample_df_compare)
    assert report.compare_df is not None
    assert report.compare_stats is not None
    # Check that PSI results exist
    if report.compare_stats and "psi" in report.compare_stats:
        assert isinstance(report.compare_stats["psi"], dict)


def test_report_null_cols(messy_df):
    """Test null columns property."""
    report = glance(messy_df)
    # Should identify columns with >30% nulls
    assert isinstance(report.null_cols, list)


def test_report_narrative(sample_df):
    """Test narrative generation."""
    report = glance(sample_df)
    narrative = report.narrative()
    assert isinstance(narrative, str)
    assert 'rows' in narrative.lower() or 'columns' in narrative.lower()


def test_cli_import():
    """Test that CLI module can be imported."""
    try:
        from edaio import cli # type: ignore
        assert cli is not None
    except ImportError:
        pytest.skip("CLI module not available")


if __name__ == "__main__":
    pytest.main([__file__, "-v"])