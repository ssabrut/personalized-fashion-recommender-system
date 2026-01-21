import numpy as np
import tensorflow as tf
import tensorflow_recommenders as tfrs
from keras import Model, Sequential, layers
from langchain_community.embeddings import DeepInfraEmbeddings

from core.config import settings

class QueryTowerFactory:
    def __init__(self, dataset: "TwoTowerDataset") -> None:
        self._dataset = dataset

    def build(self, embed_dim: int = settings.TWO_TOWER_MODEL_EMBEDDING_SIZE) -> "QueryTower":
        return QueryTower(
            emb_dim=embed_dim,
            llm_output_dim=4096
        )

class QueryTower(Model):
    def __init__(self, emb_dim: int, llm_output_dim: int = 4096, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)

        self.user_projection = Sequential([
            layers.Dense(llm_output_dim, activation="relu"),
            layers.Dense(emb_dim)
        ])

        self.normalized_age = layers.Normalization(axis=None)
        self.fnn = Sequential(
            [
                layers.Dense(emb_dim, activation="relu"),
                layers.Dense(emb_dim)
            ]
        )

    def call(self, inputs):
        user_vec = inputs["user_vector"]
        user_embedding = self.user_projection(user_vec)

        concatendated_inputs = tf.concat(
            [
                user_embedding,
                tf.reshape(self.normalized_age(inputs["age"]), (-1,1)),
                tf.reshape(inputs["month_sin"], (-1,1)),
                tf.reshape(inputs["month_cose"], (-1,1))
            ],
            axis=1
        )

        outputs = self.fnn(concatendated_inputs)
        return outputs

class ItemTowerFactory:
    def __init__(self, dataset: "TwoTowerDataset") -> None:
        self._dataset = dataset

    def build(self, embed_dim: int = settings.TWO_TOWER_MODEL_EMBEDDING_SIZE) -> "ItemTower":
        return ItemTower(
            item_ids=self._dataset.properties["item_ids"],
            garment_groups=self._dataset.properties["garment_groups"],
            index_groups=self._dataset.properties["index_groups"],
            emb_dim=embed_dim
        )

class ItemTower(Model):
    def __init__(self, item_ids: list, garment_groups: list, index_groups: list, emb_dim: int, *args, **kwargs):
        super().__init__(*args, **kwargs)

        self.garment_groups = garment_groups
        self.index_groups = index_groups

        self.item_embedding = Sequential([
            layers.StringLookup(vocabulary=item_ids, mask_token=None),
            layers.Embedding(
                len(item_ids) + 1,
                emb_dim
            )
        ])

        self.garment_group_tokenizer = layers.StringLookup(
            vocabulary=garment_groups, mask_token=None
        )

        self.index_group_tokenizer = layers.StringLookup(
            vocabulary=index_groups, mask_token=None
        )

        self.fnn = Sequential(
            [
                layers.Dense(emb_dim, activation="relu"),
                layers.Dense(emb_dim)
            ]
        )

    def call(self, inputs):
        garment_group_embedding = tf.one_hot(
            self.garment_group_tokenizer(inputs["garment_group_name"]),
            len(self.garment_groups)
        )

        index_group_embedding = tf.one_hot(
            self.index_group_tokenizer(inputs["index_group_name"]),
            len(self.index_groups)
        )

        concatenated_inputs = tf.concat(
            [
                self.item_embedding(inputs["article_id"]),
                garment_group_embedding,
                index_group_embedding
            ],
            axis=1
        )

        outputs = self.fnn(concatenated_inputs)
        return outputs

class TwoTowerModel(Model):
    def __init__(self,
                 query_model: QueryTower,
                 item_model: ItemTower,
                 item_ds: tf.data.Dataset,
                 batch_size: int,
                 *args, 
                 **kwargs
                 ) -> None:
        super().__init__(*args, **kwargs)
        self.query_model = query_model
        self.item_model = item_model
        self.task = tfrs.tasks.Retrieval(
            metrics=tfrs.metrics.FactorizedTopK(
                candidates=item_ds.batch(batch_size).map(self.item_model)
            )
        )

    def train_step(self, batch) -> tf.Tensor:
        with tf.GradientTape() as tape:
            user_embeddings = self.query_model(batch)
            item_embeddings = self.item_model(batch)
            loss = self.task(
                user_embeddings,
                item_embeddings,
                compute_metrics=False
            )

            regularization_loss = sum(self.losses)
            total_loss = loss + regularization_loss

        gradients = tape.gradient(total_loss, self.trainable_variables)
        self.optimizer.apply_gradients(zip(gradients, self.trainable_variables))

        metrics = {
            "loss": loss,
            "regularization_loss": regularization_loss,
            "total_loss": total_loss
        }

        return metrics

    def test_step(self, batch) -> tf.Tensor:
        user_embeddings = self.query_model(batch)
        item_embeddings = self.item_model(batch)

        loss = self.task(
            user_embeddings,
            item_embeddings,
            compute_metrics=False
        )

        regularization_loss = sum(self.losses)
        total_loss = loss + regularization_loss
        metrics = {metric.name: metric.result() for metric in self.metrics}
        metrics["loss"] = loss
        metrics["regularization_loss"] = regularization_loss
        metrics["total_loss"] = total_loss
        return metrics

class TwoTowerDataset:
    def __init__(self, feature_view, batch_size: int) -> None:
        self._feature_view = feature_view
        self._batch_size = batch_size
        self._properties: dict | None = None

        self.llm_embedder = DeepInfraEmbeddings(model_id="Qwen/Qwen3-Embedding-8B")

    @property
    def query_features(self) -> list[str]:
        return ["customer_id", "age", "month_sin", "month_cos", "user_vector"]

    @property
    def candidate_features(self) -> list[str]:
        return [
            "article_id",
            "garment_group_name",
            "index_group_name",
        ]

    @property
    def properties(self) -> dict:
        assert self._properties is not None, "Call get_train_val_split() first"
        return self._properties

    def get_items_subset(self):
        item_df = self.properties["train_df"][self.candidate_features]
        item_df.drop_duplicates(subset="article_id", inplace=True)
        item_ds = self.df_to_ds(item_df)
        return item_ds

    def _generate_user_strings(self, df):
        """
        Creates a descriptive string for the LLM to embed.
        Combine available features or look up user bio/metadata here.
        """
        return df.apply(
            lambda x: f"Customer {x["customer_id"]} Age: {x["age"]}", axis=1
        ).tolist()

    def _inject_deepinfra_embeddings(self, df):
        unique_users = df[["customer_id", "age"]].drop_duplicates(subset=["customer_id"])
        text_data = self._generate_user_strings(unique_users)
        vectors = self.llm_embedder.embed_documents(text_data)
        user_id_to_vec = dict(zip(unique_users["customer_id"], vectors))
        df["user_vector"] = df["customer_id"].map(user_id_to_vec)
        return df

    def get_train_val_split(self):
        train_df, val_df, test_df, _, _, _ = (
            self._feature_view.train_validation_test_split(
                validation_size=settings.TWO_TOWER_DATASET_VALIDATON_SPLIT_SIZE,
                test_size=settings.TWO_TOWER_DATASET_TEST_SPLIT_SIZE,
                description="Retrieval dataset splits",
            )
        )

        train_df = self._inject_deepinfra_embeddings(train_df)
        val_df = self._inject_deepinfra_embeddings(val_df)

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

    def df_to_ds(self, df):
        data_dict = {col: df[col].values for col in df.columns if col != "user_vector"}
        if "user_vector" in df.columns:
            vector_array = np.stack(df["user_vector"].values).astype("float32")
            data_dict["user_vector"] = vector_array

        return tf.data.Dataset.from_tensor_slices(data_dict)