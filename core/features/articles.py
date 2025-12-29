import contextlib
import io
import sys
from typing import Any

import polars as pl
from langchain_community.embeddings import DeepInfraEmbeddings
from tqdm import tqdm


def get_article_id(df: pl.DataFrame) -> pl.Series:
    return df["article_id"].cast(pl.Utf8)


def create_prod_name_length(df: pl.DataFrame) -> pl.Series:
    return df["prod_name"].str.len_chars()


def create_article_description(row: pl.Series) -> str:
    description = f"{row['prod_name']} - {row['product_type_name']} in {row['product_group_name']}"
    description += f"\nAppearance: {row['graphical_appearance_name']}"
    description += f"\nColor: {row['perceived_colour_value_name']} {row['perceived_colour_master_name']} ({row['colour_group_name']})"
    description += f"\nCategory: {row['index_group_name']} - {row['section_name']} - {row['garment_group_name']}"

    if row["detail_desc"]:
        description += f"\nDetails: {row['detail_desc']}"

    return description


def get_image_url(article_id: Any) -> str:
    url_start = "https://repo.hops.works/dev/jdowling/h-and-m/images/0"
    article_id_str = str(article_id)
    folder = article_id_str[:2]
    image_name = article_id_str
    return f"{url_start}{folder}/0{image_name}.jpg"


def compute_features_articles(df: pl.DataFrame) -> pl.DataFrame:
    df = df.with_columns(
        [
            get_article_id(df).alias("article_id"),
            create_prod_name_length(df).alias("prod_name_length"),
            pl.struct(df.columns)
            .map_elements(create_article_description)
            .alias("article_description"),
        ]
    )

    df = df.with_columns(image_url=pl.col("article_id").map_elements(get_image_url))
    df = df.select([col for col in df.columns if not df[col].is_null().any()])

    columns_to_drop = ["detail_desc", "detail_desc_length"]
    existing_columns = df.columns
    columns_to_keep = [col for col in existing_columns if col not in columns_to_drop]
    return df.select(columns_to_keep)


def generate_embedding_for_dataframe(
    df: pl.DataFrame, text_column: str, model: DeepInfraEmbeddings, batch_size: int = 32
) -> pl.DataFrame:
    @contextlib.contextmanager
    def suppress_stdout():
        new_stdout = io.StringIO()
        old_stdout = sys.stdout
        sys.stdout = new_stdout

        try:
            yield new_stdout
        finally:
            sys.stdout = old_stdout

    total_rows = len(df)
    pbar = tqdm(total=total_rows, desc="Generating embeddings")
    texts = df[text_column].to_list()

    all_embeddings = []
    for i in range(0, len(texts), batch_size):
        batch_texts = texts[i : i + batch_size]
        with suppress_stdout():
            batch_embeddings = model.embed_documents(batch_texts)
        all_embeddings.extend(batch_embeddings)
        pbar.update(len(batch_texts))

    df_with_embeddings = df.with_columns(embeddings=pl.Series(all_embeddings))
    pbar.close()
    return df_with_embeddings
