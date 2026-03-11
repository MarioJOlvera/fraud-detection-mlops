import pandas as pd

from src.data import basic_cleaning


def test_basic_cleaning_removes_duplicates():
    df = pd.DataFrame({
        "A": [1, 1, 2],
        "B": [10, 10, 20]
    })

    cleaned = basic_cleaning(df)

    assert len(cleaned) == 2
