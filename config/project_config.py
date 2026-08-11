import os
import yaml
from pydantic import BaseModel, field_validator

class PathsConfig(BaseModel):
    application_test : str
    bureau_balance : str
    bureau : str
    credit_card_balance : str
    installments_payments : str
    pos_cash_balance : str
    previous_application : str
    output_aggregation : str

    @field_validator("application_test", "bureau_balance", "bureau", "credit_card_balance",
                     "installments_payments", "pos_cash_balance", "previous_application",
                      "output_aggregation", mode="after")
    @classmethod
    def make_absolute_path(cls, value: str) -> str:
        if not os.path.isabs(value):
            return os.path.join(os.getcwd(), value)
        return value

class AppConfig(BaseModel):
    paths : PathsConfig

def load_config(config_file: str = "config.yml") -> AppConfig:
    with open(config_file, "r", encoding="utf-8") as file:
        raw_data = yaml.safe_load(file)

    return AppConfig.model_validate(raw_data)

App_config = load_config()