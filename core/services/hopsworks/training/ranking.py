from catboost import CatBoostClassifier, Pool
from sklearn.metrics import classification_report, precision_recall_fscore_support

from core.config import settings

class RankingModelFactory:
    @classmethod
    def build(cls) -> CatBoostClassifier:
        return CatBoostClassifier(
            learning_rate=settings.RANKING_LEARNING_RATE,
            iterations=settings.RANKING_ITERATIONS,
            depth=10,
            scale_pos_weight=settings.RANKING_SCALE_POS_WEIGHT,
            early_stopping_rounds=settings.RANKING_EARLY_STOPPING_ROUNDS,
            use_best_model=True
        )

