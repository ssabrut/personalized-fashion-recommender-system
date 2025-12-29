import polars as pl


def compute_ranking_dataset(trans_fg, articles_fg, customers_fg) -> pl.DataFrame:
    # read data from feature groups
    trans_df = trans_fg.select(["article_id", "customer_id"]).read(
        dataframe_type="polars"
    )

    articles_df = articles_fg.select_except(
        ["article_description", "embeddings", "image_url"]
    ).read(dataframe_type="polars")

    customers_df = customers_fg.select(["customer_id", "age"]).read(
        dataframe_type="polars"
    )

    # convert article_id to string in both dataframes before joining
    trans_df = trans_df.with_columns(pl.col("article_id").cast(pl.Utf8))
    articles_df = articles_df.with_columns(pl.col("article_id").cast(pl.Utf8))

    # merge
    df = trans_df.join(articles_df, on="article_id", how="left")
    df = df.join(customers_df, on="customer_id", how="left")

    # select query features
    query_features = ["customer_id", "age", "article_id"]
    df = df.select(query_features)

    # create positive pairs
    positive_pairs = df.clone()

    # calculate number of negative pairs
    n_neg = len(positive_pairs) * 10

    # create negative pair dataframe
    article_ids = (
        df.select("article_id")
        .unique()
        .sample(n=n_neg, with_replacement=True, seed=2)
        .get_column("article_id")
    )

    customer_ids = (
        df.select("customer_id")
        .sample(n=n_neg, with_replacement=True, seed=3)
        .get_column("customer_id")
    )

    other_features = df.select(["age"]).sample(n=n_neg, with_replacement=True, seed=4)

    # construct negative pairs
    negative_pairs = pl.DataFrame(
        {
            "article_id": article_ids,
            "customer_id": customer_ids,
            "age": other_features.get_column("age"),
        }
    )

    # add labels
    positive_pairs = positive_pairs.with_columns(pl.lit(1).alias("label"))
    negative_pairs = negative_pairs.with_columns(pl.lit(0).alias("label"))

    # concatenate positive and negative pairs
    ranking_df = pl.concat(
        [positive_pairs, negative_pairs.select(positive_pairs.columns)]
    )

    # process item features
    item_df = articles_fg.read(dataframe_type="polars")

    # convert article_id to string in item_df before final join
    item_df = item_df.with_columns(pl.col("article_id").cast(pl.Utf8))

    # keep unique article_ids and select columns
    item_df = item_df.unique(subset=["article_id"]).select(
        [
            "article_id",
            "product_type_name",
            "product_group_name",
            "graphical_appearance_name",
            "colour_group_name",
            "perceived_colour_value_name",
            "perceived_colour_master_name",
            "department_name",
            "index_name",
            "index_group_name",
            "section_name",
            "garment_group_name",
        ]
    )

    ranking_df = ranking_df.join(item_df, on="article_id", how="left")
    return ranking_df
