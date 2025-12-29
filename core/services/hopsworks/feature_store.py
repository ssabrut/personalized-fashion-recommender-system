import hopsworks
import pandas as pd
from hsfs import embedding

from core.config import settings
from core import constants

def get_feature_store():
    if settings.HOPSWORKS_API_KEY:
        project = hopsworks.login(
            api_key_value=settings.HOPSWORKS_API_KEY.get_secret_value()
        )
    else:
        project = hopsworks.login()

    return project, project.get_feature_store()

def create_customers_feature_group(fs, df: pd.DataFrame, online_enabled: bool = True):
    customers_fg = fs.get_or_create_feature_group(
        name="customers",
        description="Customers data including age and postal code",
        version=1,
        primary_key=["customer_id"],
        online_enabled=online_enabled
    )

    customers_fg.insert(df, wait=True)

    for desc in constants.CUSTOMER_FEATURE_DESCRIPTIONS:
        customers_fg.update_feature_description(desc["name"], desc["description"])

    return customers_fg

def create_articles_feature_group(
    fs,
    df: pd.DataFrame,
    articles_description_embedding_dim: int,
    online_enabled: bool = True
):
    embedding_index = embedding.EmbeddingIndex()
    embedding_index.add_embedding("embeddings", articles_description_embedding_dim)

    articles_fg = fs.get_or_create_feature_group(
        name="articles",
        version=1,
        description="Fashion items data including type of item, visual description and category",
        primary_key=["article_id"],
        online_enabled=online_enabled,
        features=constants.ARTICLE_FEATURE_DESCRIPTION,
        embedding_index=embedding_index
    )

    articles_fg.insert(df, wait=True)
    return articles_fg

def create_interactions_feature_group(
    fs, df: pd.DataFrame, online_enabled: bool = True
):
    interactions_fg = fs.get_or_create_feature_group(
        name="interactions",
        version=1,
        description="Customer interactions with articles inculding purchases, clicks and ignores. Used for building recommendation systems and analyzing user behavior",
        primary_key=["customer_id", "article_id"],
        online_enabled=online_enabled,
        event_time="t_dat"
    )

    interactions_fg.insert(
        df,
        wait=True
    )

    for desc in constants.INTERACTIONS_FEATURES_DESCRIPTIONS:
        interactions_fg.update_feature_description(desc["name"], desc["description"])
    return interactions_fg

def create_raking_feature_group(
    fs, df: pd.DataFrame, parents: list, online_enabled: bool = True
):
    rank_fg = fs.get_or_create_feature_group(
        name="ranking",
        version=1,
        description="Derived feature group for ranking",
        primary_key=["customer_id", "article_id"],
        parents=parents,
        online_enabled=online_enabled
    )

    rank_fg.insert(df, wait=True)
    for desc in constants.RANKING_FEATURE_DESCRIPTIONS:
        rank_fg.update_feature_description(desc["name"], desc["description"])

    return rank_fg

def create_candidate_embeddings_feature_group(
    fs, df: pd.DataFrame, online_enabled: bool = True
):
    embedding_index = embedding.EmbeddingIndex()
    embedding_index.add_embedding(
        "embeddings",
        settings.TWO_TOWER_MODEL_EMBEDDING_SIZE
    )

    candidate_embeddings_fg = fs.get_or_create_feature_group(
        name="candidate_embeddings",
        embedding_index=embedding_index,
        primary_key=["article_id"],
        version=1,
        description="Embeddings for each article",
        online_enabled=online_enabled
    )

    candidate_embeddings_fg.insert(df, wait=True)
    return candidate_embeddings_fg

def create_retrieval_feature_view(fs):
    trans_fg = fs.get_feature_group(name="transactions", version=1)
    customers_fg = fs.get_feature_group(name="customers", version=1)
    articles_fg = fs.get_feature_group(name="articles", version=1)

    selected_features = (
        trans_fg.select(
            ["customer_id", "article_id", "t_dat", "price", "month_sin", "month_cos"]
        )
        .join(
            customers_fg.select(["age", "club_member_status", "age_group"]),
            on="customer_id",
        )
        .join(
            articles_fg.select(["garment_group_name", "index_group_name"]),
            on="article_id",
        )
    )

    feature_view = fs.get_or_create_feature_view(
        name="retrieval",
        query=selected_features,
        version=1,
    )

    return feature_view


def create_ranking_feature_views(fs):
    customers_fg = fs.get_feature_group(
        name="customers",
        version=1,
    )

    articles_fg = fs.get_feature_group(
        name="articles",
        version=1,
    )

    rank_fg = fs.get_feature_group(
        name="ranking",
        version=1,
    )

    trans_fg = fs.get_feature_group(
        name="transactions",
        version=1)

    selected_features_customers = customers_fg.select_all()
    fs.get_or_create_feature_view(
        name="customers",
        query=selected_features_customers,
        version=1,
    )

    selected_features_articles = articles_fg.select_except(["embeddings"])
    fs.get_or_create_feature_view(
        name="articles",
        query=selected_features_articles,
        version=1,
    )

    # Select features
    selected_features_ranking = rank_fg.select_except(["customer_id", "article_id"]).join(trans_fg.select(["month_sin", "month_cos"]))
    feature_view_ranking = fs.get_or_create_feature_view(
        name="ranking",
        query=selected_features_ranking,
        labels=["label"],
        version=1,
    )

    return feature_view_ranking


def create_candidate_embeddings_feature_view(fs, fg):
    feature_view = fs.get_or_create_feature_view(
        name="candidate_embeddings",
        version=1,
        description="Embeddings of each article",
        query=fg.select(["article_id"]),
    )

    return feature_view