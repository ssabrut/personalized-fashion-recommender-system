import os
import joblib
from hsml.transformer import Transformer

from core.config import settings

class HopsworksRankingModel:
    deployment_name = "ranking"

    def __init__(self, model):
        self._model = model

    def save_to_local(self, output_path: str = "ranking_model.pkl"):
        joblib.dump(self._model, output_path)
        return output_path

    def register(self, mr, feature_view, X_train, metrics):
        local_model_path = self.save_to_local()

        input_example = X_train.sample().to_dict("records")

        ranking_model = mr.python.create_model(
            name="ranking_model",
            description="Ranking model that scores item candidates",
            metrics=metrics,
            input_example=input_example,
            feature_view=feature_view,
        )
        
        ranking_model.save(local_model_path)

    @classmethod
    def deploy(cls, project):
        mr = project.get_model_registry()
        dataset_api = project.get_dataset_api()

        ranking_model = mr.get_best_model(
            name="ranking_model",
            metric="fscore",
            direction="max",
        )

        uploaded_file_path = dataset_api.upload(
            str(
                settings.RECSYS_DIR / "services" / "hopsworks" / "inference" / "ranking_transformer.py"
            )
        )

        transformer_script_path = os.path.join(
            "/Projects",
            project.name,
            uploaded_file_path
        )

        uploaded_file_path = dataset_api.upload(
            str(
                settings.RECSYS_DIR / "services" / "hopsworks" / "inference" / "ranking_predictor.py"
            )
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

        ranking_deployment = ranking_model.deploy(
            name=cls.deployment_name,
            description="Deployment that search for item candidates and scores them based on customer metadata",
            script_file=predictor_script_path,
            resources={"num_instances": 0},
            transformer=ranking_transformer,
        )
        return ranking_deployment