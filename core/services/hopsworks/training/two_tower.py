import tensorflow as tf
import tensorflow_recommenders as tfrs
from keras.layers import Normalization, StringLookup
from keras import Model
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

