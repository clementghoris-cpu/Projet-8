import os

import yaml
from pydantic import BaseModel, SecretStr, field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


class PathsConfig(BaseModel):
    train_clean : str
    application_test : str
    bureau_balance : str
    bureau : str
    credit_card_balance : str
    installments_payments : str
    pos_cash_balance : str
    previous_application : str
    output_aggregation : str
    model_columns : str
    model_features_importance : str
    model : str

    @field_validator("train_clean", "application_test", "bureau_balance", "bureau", "credit_card_balance",
                     "installments_payments", "pos_cash_balance", "previous_application",
                      "output_aggregation", "model_columns", "model_features_importance", "model", mode="after")
    @classmethod
    def make_absolute_path(cls, value: str) -> str:
        if not os.path.isabs(value):
            return os.path.join(os.getcwd(), value)
        return value

class ModelConfig(BaseModel):
    version : str
    optimal_threshold : float

class AppConfig(BaseModel):
    paths : PathsConfig
    model : ModelConfig

def load_config(config_file: str = "config.yml") -> AppConfig:
    with open(config_file, "r", encoding="utf-8") as file:
        raw_data = yaml.safe_load(file)

    return AppConfig.model_validate(raw_data)

class CacheDbSettings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", env_prefix="redis_", extra="ignore")
    host : str = "localhost"
    port : int = 6379
    db : int = 0
    password : SecretStr = SecretStr("")

class DatabaseSettings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", env_prefix="postgresql_", extra="ignore")
    host : str = "localhost"
    port : int = 5432
    db : str = "postgres"
    user : str = "postgres"
    password : SecretStr = SecretStr("")

    @property
    def database_url(self) -> str:
        return (
            f"postgresql+psycopg://{self.user}:{self.password.get_secret_value()}"
            f"@{self.host}:{self.port}/{self.db}"
        )

class ApiSettings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", env_prefix="api_", extra="ignore")
    url : str = "http://localhost"
    port : int = 8000
    enable_cProfile : bool = False
    predict_route : str = "predict"
    logs_inference_route : str = "logs/inference-data"
    logs_profiling_route : str = "logs/profiling"

App_config = load_config(config_file = os.path.join(os.getcwd(), "config","config.yml"))
CacheDb_settings = CacheDbSettings()
Database_settings = DatabaseSettings()
Api_settings = ApiSettings()