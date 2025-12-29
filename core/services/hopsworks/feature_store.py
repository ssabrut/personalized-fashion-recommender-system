import hopsworks
import pandas as pd

from core.config import settings
from core import constants

def get_feature_store():
    if settings.HOPSWORKS_API_KEY:
        project = hopsworks.login(
            api_key_value=settings.HOPSWORKS_API_KEY.get_secret_value()
        )
    else:
        project = hopsworks.login()

    return project, project.get_feature_store()

def create_customers_feature_group(fs, df: pd.DataFrame, online_enabled: bool = True):
    customers_fg = fs.get_or_create_feature_group(
        name="customers",
        description="Customers data including age and postal code",
        version=1,
        primary_key=["customer_id"],
        online_enabled=online_enabled
    )

    customers_fg.insert(df, wait=True)

    for desc in constants.CUSTOMER_FEATURE_DESCRIPTIONS:
        customers_fg.update_feature_description(desc["name"], desc["description"])

    return customers_fg