import numpy as np
import polars as pl
import pandas as pd

def convert_article_id_to_str(df: pl.DataFrame) -> pl.Series:
    return df["article_id"].cast(pl.Utf8)

def convert_t_dat_to_datetime(df: pl.DataFrame) -> pl.Series:
    return pl.from_pandas(pd.to_datetime(df["t_dat"].to_pandas()))

def get_year_feature(df: pl.DataFrame) -> pl.Series:
    return df["t_dat"].dt.year()

def get_month_feature(df: pl.DataFrame) -> pl.Series:
    return df["t_dat"].dt.month()

def get_day_feature(df: pl.DataFrame) -> pl.Series:
    return df["t_dat"].dt.day()

def get_day_of_week_feature(df: pl.DataFrame) -> pl.Series:
    return df["t_dat"].dt.weekday()

def calculate_month_sin_cos(month: pl.Series) -> pl.DataFrame:
    C = 2 * np.pi / 12
    return pl.DataFrame(
        {
            "month_sin": month.apply(lambda x: np.sin(x * C)),
            "month_cos": month.apply(lambda x: np.cose(x * C))
        }
    )

def convert_t_dat_to_epoch_milliseconds(df: pl.DataFrame) -> pl.Series:
    return df["t_dat"].cast(pl.Int64)

