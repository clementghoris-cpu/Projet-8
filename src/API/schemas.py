from typing import Optional, Literal
from pydantic import BaseModel, Field
import pandas as pd
import numpy as np


class ClientApplicationInput(BaseModel):    
    '''Schéma Pydantic pour la validation des données entrantes de l'application client.'''
    
    # Identifiant & Type de contrat
    SK_ID_CURR: int = Field(..., gt=0, description="ID unique du client")
    NAME_CONTRACT_TYPE: Literal["Cash loans", "Revolving loans"] = Field(default="Cash loans", description="Type de contrat de prêt")

    # Informations personnelles & démographiques
    CODE_GENDER: Literal["F", "M"] = Field(..., description="Genre du client (F ou M)")
    FLAG_OWN_CAR: Literal["Y", "N"] = Field(default="N", description="Possède une voiture")
    FLAG_OWN_REALTY: Literal["Y", "N"] = Field(default="Y", description="Possède un bien immobilier")
    CNT_CHILDREN: int = Field(default=0, ge=0, description="Nombre d'enfants")
    DAYS_BIRTH: int = Field(..., lt=0, description="Âge en jours (valeur négative)")
    DAYS_EMPLOYED: Optional[float] = Field(default=None, description="Nombre de jours d'embauche")
    DAYS_REGISTRATION: Optional[float] = Field(default=None, description="Jours depuis changement d'enregistrement")
    DAYS_ID_PUBLISH: Optional[int] = Field(default=None, description="Jours depuis changement de pièce d'identité")
    
    NAME_INCOME_TYPE: str = Field(default="Working", description="Type de revenu")
    NAME_EDUCATION_TYPE: str = Field(default="Secondary / secondary special", description="Niveau d'éducation")
    NAME_FAMILY_STATUS: str = Field(default="Married", description="Statut familial")
    NAME_HOUSING_TYPE: str = Field(default="House / apartment", description="Type de logement")
    
    OCCUPATION_TYPE: Optional[str] = Field(default=None, description="Profession")
    ORGANIZATION_TYPE: Optional[str] = Field(default=None, description="Type d'organisation / employeur")
    CNT_FAM_MEMBERS: Optional[float] = Field(default=1.0, ge=1.0, description="Nombre de membres dans la famille")

    # Évaluation Région & Ville
    REGION_RATING_CLIENT: Optional[int] = Field(default=None, description="Évaluation de la région où vit le client (1,2,3)")
    REG_CITY_NOT_LIVE_CITY: int = Field(default=0, ge=0, le=1, description="1 si adresse de contact != adresse permanente")
    REG_CITY_NOT_WORK_CITY: int = Field(default=0, ge=0, le=1, description="1 si adresse de travail != adresse permanente")

    # Données financières (Montants)
    AMT_INCOME_TOTAL: float = Field(..., gt=0, description="Revenu total du client")
    AMT_CREDIT: float = Field(..., gt=0, description="Montant du crédit sollicité")
    AMT_ANNUITY: Optional[float] = Field(default=None, ge=0, description="Annuité du prêt")
    AMT_GOODS_PRICE: Optional[float] = Field(default=None, ge=0, description="Prix du bien financé")

    # Scores externes (EXT_SOURCE_*)
    EXT_SOURCE_1: Optional[float] = Field(default=None, ge=0.0, le=1.0)
    EXT_SOURCE_2: Optional[float] = Field(default=None, ge=0.0, le=1.0)
    EXT_SOURCE_3: Optional[float] = Field(default=None, ge=0.0, le=1.0)

    # Flags de contact
    FLAG_MOBIL: int = Field(default=0, ge=0, le=1)
    FLAG_EMP_PHONE: int = Field(default=0, ge=0, le=1)
    FLAG_WORK_PHONE: int = Field(default=0, ge=0, le=1)
    FLAG_CONT_MOBILE: int = Field(default=0, ge=0, le=1)
    FLAG_PHONE: int = Field(default=0, ge=0, le=1)
    FLAG_EMAIL: int = Field(default=0, ge=0, le=1)
    DAYS_LAST_PHONE_CHANGE: Optional[float] = Field(default=None)

    # Documents fournis
    FLAG_DOCUMENT_2: int = Field(default=0, ge=0, le=1)
    FLAG_DOCUMENT_3: int = Field(default=1, ge=0, le=1)
    FLAG_DOCUMENT_4: int = Field(default=0, ge=0, le=1)
    FLAG_DOCUMENT_5: int = Field(default=0, ge=0, le=1)

    # Entourage social (Social Circle)
    OBS_30_CNT_SOCIAL_CIRCLE: Optional[float] = Field(default=None, description="Observations 30 DPD")
    DEF_30_CNT_SOCIAL_CIRCLE: Optional[float] = Field(default=None, description="Défauts 30 DPD")
    OBS_60_CNT_SOCIAL_CIRCLE: Optional[float] = Field(default=None, description="Observations 60 DPD")
    DEF_60_CNT_SOCIAL_CIRCLE: Optional[float] = Field(default=None, description="Défauts 60 DPD")

    # Demandes auprès du bureau de crédit (AMT_REQ_CREDIT_BUREAU_*)
    AMT_REQ_CREDIT_BUREAU_HOUR: Optional[float] = Field(default=None, ge=0)
    AMT_REQ_CREDIT_BUREAU_DAY: Optional[float] = Field(default=None, ge=0)
    AMT_REQ_CREDIT_BUREAU_WEEK: Optional[float] = Field(default=None, ge=0)
    AMT_REQ_CREDIT_BUREAU_MON: Optional[float] = Field(default=None, ge=0)
    AMT_REQ_CREDIT_BUREAU_QRT: Optional[float] = Field(default=None, ge=0)
    AMT_REQ_CREDIT_BUREAU_YEAR: Optional[float] = Field(default=None, ge=0)

    # Convertisseur vers DataFrame Pandas
    def to_pandas(self) -> pd.DataFrame:
        '''
        Convertit l'objet Pydantic validé en DataFrame, 
        avec remplacement automatique des `None` par `np.nan` (format Pandas).
        '''
        data = self.model_dump()
        df = pd.DataFrame([data])
        df = df.replace({None: np.nan})
        return df