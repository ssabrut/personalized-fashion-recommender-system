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