import pandas as pd

from config.project_config import App_config
from src.data.historic.bureau_aggregation import process_bureau, process_bureau_balance
from src.data.historic.credit_card_balance_aggregation import process_credit_card
from src.data.historic.installments_payments_aggregation import process_install_payments
from src.data.historic.pos_cash_balance_aggregation import process_pos_cash_balance
from src.data.historic.previous_application_aggregation import process_previous_app

BUREAU = "bureau"
BUREAU_BALANCE = "bureau_balance"
CREDIT_CARD = "credit_card"
INSTALL_PAYMENTS = "install_payments"
POS_CASH = "posh_cash"
PREV_APP = "previous_app"

def read_historic_files() -> dict:
    '''
    Lis l'ensemble des fichiers historique et insert les données dans un dictionnaire de dataframes
    '''
    try:
        df_dict = {
            BUREAU : pd.read_csv(App_config.paths.bureau),
            BUREAU_BALANCE : pd.read_csv(App_config.paths.bureau_balance),
            CREDIT_CARD : pd.read_csv(App_config.paths.credit_card_balance),
            INSTALL_PAYMENTS : pd.read_csv(App_config.paths.installments_payments),
            POS_CASH : pd.read_csv(App_config.paths.pos_cash_balance),
            PREV_APP : pd.read_csv(App_config.paths.previous_application)
        }

        return df_dict
    
    except Exception as ex:
        print(f"Erreur lecture fichiers historiques : {ex.__traceback__}")
        raise
        

def calculate_aggregation_from_historic(df_dict : dict) -> pd.DataFrame:
    '''
    Calcul les agrégation pour chaque donnée historique et merge l'ensemble en un seul dataframe
    '''
    df_agg_bureau = process_bureau(df_dict[BUREAU])
    df_agg_bureau_balance = process_bureau_balance(df_dict[BUREAU_BALANCE], df_dict[BUREAU])
    df_agg_credit_card = process_credit_card(df_dict[CREDIT_CARD])
    df_agg_install_pay = process_install_payments(df_dict[INSTALL_PAYMENTS])
    df_agg_pos_cash = process_pos_cash_balance(df_dict[POS_CASH])
    df_agg_prev_app = process_previous_app(df_dict[PREV_APP])

    ID = "SK_ID_CURR"

    df_agg_total = df_agg_bureau.merge(df_agg_bureau_balance, on=ID, how='outer')
    df_agg_total = df_agg_total.merge(df_agg_credit_card, on=ID, how='outer')
    df_agg_total = df_agg_total.merge(df_agg_install_pay, on=ID, how='outer')
    df_agg_total = df_agg_total.merge(df_agg_pos_cash, on=ID, how='outer')
    df_agg_total = df_agg_total.merge(df_agg_prev_app, on=ID, how='outer')

    return df_agg_total


def run_historic_aggregation():
    '''
    1. Lit les différents fichiers historique pour récupérer les données sous forme de dataframe
    2. Calcul les agrégations de l'ensemble des données historique
    3. enregistre les résultats dans un fichier CSV
    '''
    try:
        print("Lecture des fichiers historiques...")
        df_dict = read_historic_files()

        print("Agrégation des données historiques...")
        print("--> Résultat agrégation :")
        df_agg = calculate_aggregation_from_historic(df_dict)

        print("Sauvegarde des données historique agrégées...")
        df_agg.to_csv(App_config.paths.output_aggregation, index=False)

        print("Historic pipeline executé avec succès!")
        # Le commentaire ajouté à droite ci-dessous est pour éviter que la librairie Ruff renvoi une erreur (utilisation d'une Exception générique) à la vérification du code
    except Exception as e:  # noqa: BLE001
        print(f"Erreur lors du calcul des agrégation des données historiques : {e}")


if __name__ == "__main__":
    run_historic_aggregation()