import polars as pl

def get_article_id(df: pl.DataFrame) -> pl.Series:
    return df["article_id"].cast(pl.Utf8)

def create_prod_name_length(df: pl.DataFrame) -> pl.Series:
    return df["prod_name"].str.len_chars()

def create_article_description(row):
    description = f"{row['prod_name']} - {row['product_type_name']} in {row['product_group_name']}"
    description += f"\nAppearance: {row['graphical_appearance_name']}"
    description += f"\nColor: {row['perceived_colour_value_name']} {row['perceived_colour_master_name']} ({row['colour_group_name']})"
    description += f"\nCategory: {row['index_group_name']} - {row['section_name']} - {row['garment_group_name']}"

    if row['detail_desc']:
        description += f"\nDetails: {row['detail_desc']}"

    return description

