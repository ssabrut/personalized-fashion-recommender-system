import os
from typing import Literal
import hopsworks
import tensorflow as tf
from hsml.transformer import Transformer
from loguru import logger

from core.config import settings
from core.services.hopsworks.training.two_tower import ItemTower, QueryTower

class QueryModelModule(tf.Module):
    def __init__(self, model: QueryTower) -> None:
        self.model = model

    @tf.function()
    def compute_embedding(self, instances):
        query_embedding = self.model(instances)

        return {
            "customer_id": instances["customer_id"],
            "month_sin": instances["month_sin"],
            "month_cos": instances["month_cos"],
            "query_emb": query_embedding,
        }

class HopsworksQueryModel:
    deployment_name = "query"

    def __init__(self, model: QueryTower) -> None:
        self.model = model

    def save_to_local(self, output_path: str = "query_model") -> str:
        instances_spec = {
            "customer_id": tf.TensorSpec(
                shape=(None,), dtype=tf.string, name="customer_id"
            ),  # Specification for customer IDs
            "month_sin": tf.TensorSpec(
                shape=(None,), dtype=tf.float64, name="month_sin"
            ),  # Specification for sine of month
            "month_cos": tf.TensorSpec(
                shape=(None,), dtype=tf.float64, name="month_cos"
            ),  # Specification for cosine of month
            "age": tf.TensorSpec(
                shape=(None,), dtype=tf.float64, name="age"
            ),  # Specification for age
        }

        query_module_module = QueryModelModule(model=self.model)
        # Get the concrete function for the query_model's compute_emb function using the specified input signatures
        inference_signatures = (
            query_module_module.compute_embedding.get_concrete_function(instances_spec)
        )

        # Save the query_model along with the concrete function signatures
        tf.saved_model.save(
            self.model,  # The model to save
            output_path,  # Path to save the model
            signatures=inference_signatures,  # Concrete function signatures to include
        )

        return output_path

    def register(self, mr, feature_view, query_df) -> None:
        local_model_path = self.save_to_local()

        # Sample a query example from the query DataFrame
        query_example = query_df.sample().to_dict("records")

        # Create a tensorflow model for the query_model in the Model Registry
        mr_query_model = mr.tensorflow.create_model(
            name="query_model",  # Name of the model
            description="Model that generates query embeddings from user and transaction features",  # Description of the model
            input_example=query_example,  # Example input for the model
            feature_view=feature_view,
        )

        # Save the query_model to the Model Registry
        mr_query_model.save(local_model_path)  # Path to save the model

    @classmethod
    def deploy(cls, ranking_model_type: Literal["ranking", "llmranking"] = "ranking"):
        # Prepare secrets used in the deployment
        project = hopsworks.login()
        cls._prepare_secrets(ranking_model_type)

        mr = project.get_model_registry()
        dataset_api = project.get_dataset_api()

        # Retrieve the 'query_model' from the Model Registry
        query_model = mr.get_model(
            name="query_model",
            version=1,
        )

        # Copy transformer file into Hopsworks File System
        uploaded_file_path = dataset_api.upload(
            str(settings.RECSYS_DIR / "services" / "hopsworks" / "inference" / "query_transformer.py"),
            "Models",
            overwrite=True,
        )

        # Construct the path to the uploaded script
        transformer_script_path = os.path.join(
            "/Projects",
            project.name,
            uploaded_file_path,
        )

        query_model_transformer = Transformer(
            script_file=transformer_script_path,
            resources={"num_instances": 0},
        )

        # Deploy the query model
        query_model_deployment = query_model.deploy(
            name=cls.deployment_name,
            description="Deployment that generates query embeddings from customer and item features using the query model",
            resources={"num_instances": 0},
            transformer=query_model_transformer,
        )

        return query_model_deployment

    @classmethod
    def _prepare_secrets(cls, ranking_model_type: Literal["ranking", "llmranking"]):
        project = hopsworks.login(
            hostname_verification=False,
            api_key_value=settings.HOPSWORKS_API_KEY.get_secret_value(),     
        )
        secrets_api = hopsworks.get_secrets_api()
        secrets = secrets_api.get_secrets()
        existing_secret_keys = [secret.name for secret in secrets]
        if "RANKING_MODEL_TYPE" in existing_secret_keys:
            secrets_api._delete(name="RANKING_MODEL_TYPE")

        secrets_api.create_secret(
            "RANKING_MODEL_TYPE",
            ranking_model_type,
            project=project.name,
        )