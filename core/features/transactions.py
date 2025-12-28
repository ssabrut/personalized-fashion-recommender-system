import polars as pl
import pandas as pd

def convert_article_id_to_str(df: pl.DataFrame) -> pl.Series:
    return df["article_id"].cast(pl.Utf8)

def convert_t_dat_to_datetime(df: pl.DataFrame) -> pl.Series:
    return pl.from_pandas(pd.to_datetime(df["t_dat"].to_pandas()))

def get_year_feature(df: pl.DataFrame) -> pl.Series:
    return df["t_dat"].dt.year()