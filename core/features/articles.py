import polars as pl

def get_article_id(df: pl.DataFrame) -> pl.Series:
    return df["article_id"].cast(pl.Utf8)

def create_prod_name_length(df: pl.DataFrame) -> pl.Series:
    return df["prod_name"].str.len_chars()

