from scripts.model_train import normalize_series
import pandas as pd
import pytest

@pytest.mark.parametrize(
    "input_series, expected_series",
    [
        #Basic test
        (
            pd.Series([1.0, 2.0, 3.0, 4.0, 5.0]),
            pd.Series([-1.264911, -0.632456, 0.000000, 0.632456, 1.264911])
        ),
         #Series with all identical values (should be NaN after normalization)
        (
            pd.Series([5, 5, 5, 5, 5]),
            pd.Series([float('nan')] * 5)
        ),
        #Series with a mix of positive and negative values
        (
            pd.Series([-3, -1, 0, 1, 3]),
            pd.Series([-1.34164, -0.44721, 0.0, 0.44721, 1.34164])
        ),
        #Series with single value (should be NaN after normalization)
        (
            pd.Series([10]),
            pd.Series([float('nan')])
        ),
        # Series with different data types
        (
            pd.Series([10.0, 20, 30.0, 40, 50]),
            pd.Series([-1.264911, -0.632456, 0.000000, 0.632456, 1.264911])
        )
    ]
)
def test_normalization(input_series,expected_series):
    result = normalize_series(input_series)
    pd.testing.assert_series_equal(result, expected_series, check_exact=False, check_dtype=False)
