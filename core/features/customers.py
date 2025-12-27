import random
import polars as pl
from typing import Dict

from core.config import CustomerDatasetSize

class DatasetSampler:
    _SIZES = {
        CustomerDatasetSize.LARGE: 50_000,
        CustomerDatasetSize.MEDIUM: 5_000,
        CustomerDatasetSize.SMALL: 1_000
    }

    def __init__(self, size: CustomerDatasetSize) -> None:
        self._size = size

    @classmethod
    def get_supported_sizes(cls) -> dict:
        return cls._SIZES

    def sample(
        self, customer_df: pl.DataFrame, transaction_df: pl.DataFrame
    ) -> Dict[str, pl.DataFrame]:
        random.seed(27)
        n_customers = self._SIZES[self._size]
        customer_df = customer_df.sample(n=n_customers)

        transaction_df = transaction_df.join(
            customer_df.select("customer_id"), on="customer_id"
        )

        return {"customers": customer_df, "transactions": transaction_df}