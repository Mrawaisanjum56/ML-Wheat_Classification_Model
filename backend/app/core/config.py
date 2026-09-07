from pydantic_settings import BaseSettings, SettingsConfigDict
from typing import List


class Settings(BaseSettings):
    APP_NAME: str = "WheatVision AI"
    APP_VERSION: str = "1.0.0"
    MODEL_PATH: str = "app/ml/artifacts/model.pkl"
    MAX_FILE_SIZE_MB: int = 5
    ALLOWED_ORIGINS: List[str] = ["*"]  # set specific domains in production

    model_config = SettingsConfigDict(env_file=".env", case_sensitive=False)


settings = Settings()