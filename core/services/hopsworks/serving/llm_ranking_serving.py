import os
import hopsworks
from hsml.transformer import Transformer

from core.config import settings

class HopsworksLLMRankingModel:
    deployment_name = "llmranking"

    @classmethod
    def register(cls, mr):
        local_model_path = str(
            settings.RECSYS_DIR / "core" / "services" / "hopsworks" / "inference" / "llm_ranking_predictor.py"
        )

        ranking_model = mr.python.create_model(
            name="llm_ranking_model",
            description="LLM Ranking model that scores item candidates"
        )

        ranking_model.save(local_model_path)

    @classmethod
    def deploy(cls):
        cls._prepare_secrets()

        project = hopsworks.login()
        cls._prepare_environment(project)
        mr = project.get_model_registry()
        dataset_api = project.get_dataset_api()

        ranking_model = mr.get_model(name="llm_ranking_model")

        # transformer file
        uploaded_file_path = dataset_api.upload(
            str(
                settings.RECSYS_DIR / "services" / "hopsworks" / "inference" / "ranking_transformer.py"
            ),
            "Resources",
            overwrite=True
        )

        transformer_script_path = os.path.join(
            "/Projects",
            project.name,
            uploaded_file_path
        )

        # llm predictor
        uploaded_file_path = dataset_api.upload(
            str(settings.RECSYS_DIR / "services" / "hopsworks" / "inference" / "llm_ranking_predictor.py"),
            "Resources",
            overwrite=True
        )

        predictor_script_path = os.path.join(
            "/Projects",
            project.name,
            uploaded_file_path
        )

        ranking_transformer = Transformer(
            script_file=transformer_script_path,
            resources={"num_instances": 0}
        )

        # deploy ranking model
        ranking_deployment = ranking_model.deploy(
            name=cls.deployment_name,
            description="Deployment that search for item candidates and scores them based on customer metadata using GPT-OSS",
            script_file=predictor_script_path,
            resources={"num_instances": 0},
            transformer=ranking_transformer,
            environment=settings.CUSTOM_HOPSWORKS_INFERENCE_ENV
        )

        return ranking_deployment

    @classmethod
    def _prepare_environment(cls, project):
        dataset_api = project.get_dataset_api()
        requirements_path = dataset_api.upload(
            str(
                settings.RECSYS_DIR
                / "services"
                / "hopsworks"
                / "requirements.txt"
            ),
            "Resources",
            overwrite=True
        )

        env_api = project.get_environment_api()
        envs = env_api.get_environments()
        existing_envs = [env.name for env in envs]

        if settings.CUSTOM_HOPSWORKS_INFERENCE_ENV in existing_envs:
            env = env_api.get_environment(settings.CUSTOM_HOPSWORKS_INFERENCE_ENV)
        else:
            env = env_api.create_environment(
                name=settings.CUSTOM_HOPSWORKS_INFERENCE_ENV,
                base_environment_name="pandas-inference-pipeline"
            )

        env.install_requirements(requirements_path)

    @classmethod
    def _prepare_secrets(cls):
        if not settings.DEEPINFRA_API_TOKEN:
            raise ValueError("Missing required secret: 'DEEPINFRA_API_TOKEN'. Please ensure it is set in the .env file or config.py")

    project = hopsworks.login(
        api_key_value=settings.HOPSWORKS_API_KEY.get_secret_value()
    )

    secrets_api = hopsworks.get_secret_api()
    secrets = secrets_api.get_secrets()
    existing_secret_keys = [secret.name for secret in secrets]

    if "DEEPINFRA_API_TOKEN" in existing_secret_keys:
        secrets_api._delete(name="DEEPINFRA_API_TOKEN")

    secrets_api.create_secret(
        "DEEPINFRA_API_TOKEN",
        settings.DEEPINFRA_API_TOKEN.get_secret_value(),
        project=project.name
    )