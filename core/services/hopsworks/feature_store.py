import hopsworks

from core.config import settings

def get_feature_store():
    if settings.HOPSWORKS_API_KEY:
        project = hopsworks.login(
            api_key_value=settings.HOPSWORKS_API_KEY.get_secret_value()
        )
    else:
        project = hopsworks.login()

    return project, project.get_feature_store()

