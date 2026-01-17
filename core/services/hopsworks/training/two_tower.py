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
        pass