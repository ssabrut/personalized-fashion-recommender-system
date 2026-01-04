import tensorflow as tf
import tensorflow_recommenders as tfrs
from keras.layers import Normalization, StringLookup

from core.config import settings

class TwoTowerDataset:
    def __init__(self, feature_view, batch_size: int) -> None:
        self._feature_view = feature_view
        self._batch_size = batch_size
        self._properties: dict | None

    @property
    def query_features(self) -> list[str]:
        return ["customer_id", "age", "month_sin", "month_cos"]

    @property
    def candidate_features(self) -> list[str]:
        return ["article_id", "garment_group_name", "index_group_name"]

    @property
    def properties(self) -> dict:
        assert self._properties is not None, "Call get_train_val_split() first."

        return self._properties

    def df_to_ds(self, df):
        return tf.data.Dataset.from_tensor_slices({col: df[col] for col in df})

    def get_items_subset(self):
        item_df = self.properties["train_df"][self.candidate_features]
        item_df.drop_duplicates(subset="article_id", inplace=True)
        item_ds = self.df_to_ds(item_df)
        return item_ds

    def get_train_val_split(self):
        train_df, val_df, test_df, _, _, _ = (
            self._feature_view.train_validation_test_split(
                validation_size=settings.TWO_TOWER_DATASET_VALIDATON_SPLIT_SIZE,
                test_size=settings.TWO_TOWER_DATASET_TEST_SPLIT_SIZE,
                description="Retrieval dataset splits"
            )
        )

        train_ds = (
            self.df_to_ds(train_df)
            .batch(self._batch_size)
            .cache()
            .shuffle(self._batch_size * 10)
        )

        val_ds = self.df_to_ds(val_df).batch(self._batch_size).cache()

        self._properties = {
            "train_df": train_df,
            "val_df": val_df,
            "query_df": train_df[self.query_features],
            "item_df": train_df[self.candidate_features],
            "user_ids": train_df["customer_id"].unique().tolist(),
            "item_ids": train_df["article_id"].unique().tolist(),
            "garment_groups": train_df["garment_group_name"].unique().tolist(),
            "index_groups": train_df["index_group_name"].unique().tolist(),
        }

        return train_ds, val_ds