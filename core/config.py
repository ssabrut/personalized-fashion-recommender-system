from enum import Enum
from pathlib import Path
from typing import ClassVar, Literal

from pydantic import Field, SecretStr
from pydantic_settings import BaseSettings, SettingsConfigDict


class CustomerDatasetSize(Enum):
    LARGE: str = "LARGE"
    MEDIUM: str = "MEDIUM"
    SMALL: str = "SMALL"


class DefaultSettings(BaseSettings):
    model_config: ClassVar[SettingsConfigDict] = SettingsConfigDict(
        env_file=".env", extra="ignore", frozen=True, env_nested_delimiter="__"
    )


class Settings(DefaultSettings):
    app_version: str = "0.1.0"
    debug: bool = True
    environment: str = "development"
    service_name: str = "fashion-recommender-system"

    RECSYS_DIR: Path = Path(__file__).parent

    # Hopsworks config
    HOPSWORKS_API_KEY: SecretStr | None = Field(..., env="HOPSWORKS_API_KEY")

    # DeepInfra config
    DEEPINFRA_API_KEY: SecretStr | None = Field(..., env="DEEPINFRA_API_KEY")
    DEEPINFRA_MODEL_ID: str = "openai/gpt-oss-120b"

    CUSTOMER_DATA_SIZE: CustomerDatasetSize = CustomerDatasetSize.LARGE
    FEATURE_EMBEDDING_MODEL_ID: str = "Qwen/Qwen3-Embedding-8B"

    # Training
    TWO_TOWER_MODEL_EMBEDDING_SIZE: int = 16
    TWO_TOWER_MODEL_BATCH_SIZE: int = 2048
    TWO_TOWER_NUM_EPOCHS: int = 10
    TWO_TOWER_WEIGHT_DECAY: float = 0.001
    TWO_TOWER_LEARNING_RATE: float = 0.01
    TWO_TOWER_DATASET_VALIDATON_SPLIT_SIZE: float = 0.1
    TWO_TOWER_DATASET_TEST_SPLIT_SIZE: float = 0.1

    RANKING_DATASET_VALIDATON_SPLIT_SIZE: float = 0.1
    RANKING_LEARNING_RATE: float = 0.2
    RANKING_ITERATIONS: int = 100
    RANKING_SCALE_POS_WEIGHT: int = 10
    RANKING_EARLY_STOPPING_ROUNDS: int = 5

    # Inference
    RANKING_MODEL_TYPE: Literal["ranking", "llmranking"] = "ranking"
    CUSTOM_HOPSWORKS_INFERENCE_ENV: str = "custom_env_name"


def get_settings() -> Settings:
    return Settings()
