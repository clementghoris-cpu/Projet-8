import os
import requests
import pandas as pd
import gradio as gr
from config.project_config import Api_settings, App_config


API_URL = f"{Api_settings.url}:{Api_settings.port}/{Api_settings.predict_route}"
TEST_CSV_PATH = App_config.paths.application_test

def load_test_clients(filepath, n=10):
    try:
        df = pd.read_csv(filepath)
        df_sample = df.sample(n=n, random_state=None)
        clients_dict = df_sample.set_index('SK_ID_CURR').to_dict(orient='index')
        print(f"{len(clients_dict)} clients de test chargés avec succès.")
        return clients_dict
    except Exception as e:
        print(f"Impossible de charger {filepath} ({e}). Utilisation d'un client par défaut.")
        return {
            100002: {
                "NAME_CONTRACT_TYPE": "Cash loans", "CODE_GENDER": "M", "FLAG_OWN_CAR": "N",
                "FLAG_OWN_REALTY": "Y", "CNT_CHILDREN": 0, "DAYS_BIRTH": -12000, "DAYS_EMPLOYED": -1000,
                "DAYS_REGISTRATION": -1000, "DAYS_ID_PUBLISH": -1000, "NAME_INCOME_TYPE": "Working",
                "NAME_EDUCATION_TYPE": "Secondary / secondary special", "NAME_FAMILY_STATUS": "Single / not married",
                "NAME_HOUSING_TYPE": "House / apartment", "OCCUPATION_TYPE": "Laborers",
                "ORGANIZATION_TYPE": "Business Entity Type 3", "CNT_FAM_MEMBERS": 1.0,
                "REGION_RATING_CLIENT": 2, "REG_CITY_NOT_LIVE_CITY": 0, "REG_CITY_NOT_WORK_CITY": 0,
                "AMT_INCOME_TOTAL": 202500.0, "AMT_CREDIT": 406597.5, "AMT_ANNUITY": 24700.5,
                "AMT_GOODS_PRICE": 351000.0, "EXT_SOURCE_1": 0.083, "EXT_SOURCE_2": 0.262, "EXT_SOURCE_3": 0.139,
                "FLAG_MOBIL": 1, "FLAG_EMP_PHONE": 1, "FLAG_WORK_PHONE": 0, "FLAG_CONT_MOBILE": 1,
                "FLAG_PHONE": 1, "FLAG_EMAIL": 0, "DAYS_LAST_PHONE_CHANGE": -1134.0, "FLAG_DOCUMENT_2": 0,
                "FLAG_DOCUMENT_3": 1, "FLAG_DOCUMENT_4": 0, "FLAG_DOCUMENT_5": 0,
                "OBS_30_CNT_SOCIAL_CIRCLE": 2.0, "DEF_30_CNT_SOCIAL_CIRCLE": 2.0,
                "OBS_60_CNT_SOCIAL_CIRCLE": 2.0, "DEF_60_CNT_SOCIAL_CIRCLE": 2.0,
                "AMT_REQ_CREDIT_BUREAU_HOUR": 0.0, "AMT_REQ_CREDIT_BUREAU_DAY": 0.0,
                "AMT_REQ_CREDIT_BUREAU_WEEK": 0.0, "AMT_REQ_CREDIT_BUREAU_MON": 0.0,
                "AMT_REQ_CREDIT_BUREAU_QRT": 0.0, "AMT_REQ_CREDIT_BUREAU_YEAR": 1.0
            }
        }


CLIENTS_MEMORY = load_test_clients(TEST_CSV_PATH, n=10)
CLIENT_IDS = [str(k) for k in CLIENTS_MEMORY.keys()]

def populate_form(sk_id):
    try:
        sk_id_int = int(sk_id)
    except (ValueError, TypeError):
        sk_id_int = None

    if sk_id_int not in CLIENTS_MEMORY:
        return [gr.skip()] * len(all_form_inputs)

    data = CLIENTS_MEMORY[sk_id_int]

    def safe_get(key, default):
        val = data.get(key)
        if pd.isna(val) or val is None:
            return default
        return val

    # Conversions utiles pour les sliders / champs
    days_birth = safe_get("DAYS_BIRTH", -12000)
    age_years = int(-days_birth / 365)

    days_emp = safe_get("DAYS_EMPLOYED", 0)
    years_emp = 0.0 if days_emp == 365243 else round(-float(days_emp) / 365.0, 1)

    return (
        safe_get("NAME_CONTRACT_TYPE", "Cash loans"),
        safe_get("CODE_GENDER", "F"),
        safe_get("FLAG_OWN_CAR", "N"),
        safe_get("FLAG_OWN_REALTY", "Y"),
        int(safe_get("CNT_CHILDREN", 0)),
        age_years,
        years_emp,
        round(abs(float(safe_get("DAYS_REGISTRATION", -1000))), 1),
        round(abs(float(safe_get("DAYS_ID_PUBLISH", -1000))), 1),
        str(safe_get("NAME_INCOME_TYPE", "Working")),
        str(safe_get("NAME_EDUCATION_TYPE", "Secondary / secondary special")),
        str(safe_get("NAME_FAMILY_STATUS", "Married")),
        str(safe_get("NAME_HOUSING_TYPE", "House / apartment")),
        str(safe_get("OCCUPATION_TYPE", "Laborers")),
        str(safe_get("ORGANIZATION_TYPE", "Business Entity Type 3")),
        float(safe_get("CNT_FAM_MEMBERS", 1.0)),
        int(safe_get("REGION_RATING_CLIENT", 2)),
        int(safe_get("REG_CITY_NOT_LIVE_CITY", 0)),
        int(safe_get("REG_CITY_NOT_WORK_CITY", 0)),
        float(safe_get("AMT_INCOME_TOTAL", 150000.0)),
        float(safe_get("AMT_CREDIT", 450000.0)),
        float(safe_get("AMT_ANNUITY", 22500.0)),
        float(safe_get("AMT_GOODS_PRICE", 450000.0)),
        float(safe_get("EXT_SOURCE_1", 0.5)),
        float(safe_get("EXT_SOURCE_2", 0.5)),
        float(safe_get("EXT_SOURCE_3", 0.5)),
        int(safe_get("FLAG_MOBIL", 1)),
        int(safe_get("FLAG_EMP_PHONE", 1)),
        int(safe_get("FLAG_WORK_PHONE", 0)),
        int(safe_get("FLAG_CONT_MOBILE", 1)),
        int(safe_get("FLAG_PHONE", 0)),
        int(safe_get("FLAG_EMAIL", 0)),
        round(abs(float(safe_get("DAYS_LAST_PHONE_CHANGE", -100))), 1),
        int(safe_get("FLAG_DOCUMENT_2", 0)),
        int(safe_get("FLAG_DOCUMENT_3", 1)),
        int(safe_get("FLAG_DOCUMENT_4", 0)),
        int(safe_get("FLAG_DOCUMENT_5", 0)),
        float(safe_get("OBS_30_CNT_SOCIAL_CIRCLE", 0.0)),
        float(safe_get("DEF_30_CNT_SOCIAL_CIRCLE", 0.0)),
        float(safe_get("OBS_60_CNT_SOCIAL_CIRCLE", 0.0)),
        float(safe_get("DEF_60_CNT_SOCIAL_CIRCLE", 0.0)),
        float(safe_get("AMT_REQ_CREDIT_BUREAU_HOUR", 0.0)),
        float(safe_get("AMT_REQ_CREDIT_BUREAU_DAY", 0.0)),
        float(safe_get("AMT_REQ_CREDIT_BUREAU_WEEK", 0.0)),
        float(safe_get("AMT_REQ_CREDIT_BUREAU_MON", 0.0)),
        float(safe_get("AMT_REQ_CREDIT_BUREAU_QRT", 0.0)),
        float(safe_get("AMT_REQ_CREDIT_BUREAU_YEAR", 0.0))
    )

def send_prediction_request(
    sk_id_curr, name_contract_type, code_gender, flag_own_car, flag_own_realty,
    cnt_children, age_years, years_employed, days_registration, days_id_publish,
    name_income_type, name_education_type, name_family_status, name_housing_type,
    occupation_type, organization_type, cnt_fam_members, region_rating_client,
    reg_city_not_live_city, reg_city_not_work_city, amt_income_total, amt_credit,
    amt_annuity, amt_goods_price, ext_source_1, ext_source_2, ext_source_3,
    flag_mobil, flag_emp_phone, flag_work_phone, flag_cont_mobile, flag_phone,
    flag_email, days_last_phone_change, flag_document_2, flag_document_3,
    flag_document_4, flag_document_5, obs_30_cnt_social_circle, def_30_cnt_social_circle,
    obs_60_cnt_social_circle, def_60_cnt_social_circle, amt_req_credit_bureau_hour,
    amt_req_credit_bureau_day, amt_req_credit_bureau_week, amt_req_credit_bureau_mon,
    amt_req_credit_bureau_qrt, amt_req_credit_bureau_year
):
    try:
        sk_id_val = int(sk_id_curr) if sk_id_curr is not None else 999999
        # Reconversion exacte au format attendu par Pydantic (Jours négatifs)
        days_birth = -int(age_years * 365)
        days_employed = -float(years_employed * 365) if years_employed is not None else None
        days_reg = -float(days_registration) if days_registration is not None else None
        days_id = -int(days_id_publish) if days_id_publish is not None else None
        days_phone = -float(days_last_phone_change) if days_last_phone_change is not None else None

        payload = {
            "SK_ID_CURR": sk_id_val,
            "NAME_CONTRACT_TYPE": name_contract_type,
            "CODE_GENDER": code_gender,
            "FLAG_OWN_CAR": flag_own_car,
            "FLAG_OWN_REALTY": flag_own_realty,
            "CNT_CHILDREN": int(cnt_children),
            "DAYS_BIRTH": days_birth,
            "DAYS_EMPLOYED": days_employed,
            "DAYS_REGISTRATION": days_reg,
            "DAYS_ID_PUBLISH": days_id,
            "NAME_INCOME_TYPE": name_income_type,
            "NAME_EDUCATION_TYPE": name_education_type,
            "NAME_FAMILY_STATUS": name_family_status,
            "NAME_HOUSING_TYPE": name_housing_type,
            "OCCUPATION_TYPE": occupation_type if occupation_type else None,
            "ORGANIZATION_TYPE": organization_type if organization_type else None,
            "CNT_FAM_MEMBERS": float(cnt_fam_members),
            "REGION_RATING_CLIENT": int(region_rating_client),
            "REG_CITY_NOT_LIVE_CITY": int(reg_city_not_live_city),
            "REG_CITY_NOT_WORK_CITY": int(reg_city_not_work_city),
            "AMT_INCOME_TOTAL": float(amt_income_total),
            "AMT_CREDIT": float(amt_credit),
            "AMT_ANNUITY": float(amt_annuity) if amt_annuity is not None else None,
            "AMT_GOODS_PRICE": float(amt_goods_price) if amt_goods_price is not None else None,
            "EXT_SOURCE_1": float(ext_source_1) if ext_source_1 is not None else None,
            "EXT_SOURCE_2": float(ext_source_2) if ext_source_2 is not None else None,
            "EXT_SOURCE_3": float(ext_source_3) if ext_source_3 is not None else None,
            "FLAG_MOBIL": int(flag_mobil),
            "FLAG_EMP_PHONE": int(flag_emp_phone),
            "FLAG_WORK_PHONE": int(flag_work_phone),
            "FLAG_CONT_MOBILE": int(flag_cont_mobile),
            "FLAG_PHONE": int(flag_phone),
            "FLAG_EMAIL": int(flag_email),
            "DAYS_LAST_PHONE_CHANGE": days_phone,
            "FLAG_DOCUMENT_2": int(flag_document_2),
            "FLAG_DOCUMENT_3": int(flag_document_3),
            "FLAG_DOCUMENT_4": int(flag_document_4),
            "FLAG_DOCUMENT_5": int(flag_document_5),
            "OBS_30_CNT_SOCIAL_CIRCLE": float(obs_30_cnt_social_circle) if obs_30_cnt_social_circle is not None else None,
            "DEF_30_CNT_SOCIAL_CIRCLE": float(def_30_cnt_social_circle) if def_30_cnt_social_circle is not None else None,
            "OBS_60_CNT_SOCIAL_CIRCLE": float(obs_60_cnt_social_circle) if obs_60_cnt_social_circle is not None else None,
            "DEF_60_CNT_SOCIAL_CIRCLE": float(def_60_cnt_social_circle) if def_60_cnt_social_circle is not None else None,
            "AMT_REQ_CREDIT_BUREAU_HOUR": float(amt_req_credit_bureau_hour) if amt_req_credit_bureau_hour is not None else None,
            "AMT_REQ_CREDIT_BUREAU_DAY": float(amt_req_credit_bureau_day) if amt_req_credit_bureau_day is not None else None,
            "AMT_REQ_CREDIT_BUREAU_WEEK": float(amt_req_credit_bureau_week) if amt_req_credit_bureau_week is not None else None,
            "AMT_REQ_CREDIT_BUREAU_MON": float(amt_req_credit_bureau_mon) if amt_req_credit_bureau_mon is not None else None,
            "AMT_REQ_CREDIT_BUREAU_QRT": float(amt_req_credit_bureau_qrt) if amt_req_credit_bureau_qrt is not None else None,
            "AMT_REQ_CREDIT_BUREAU_YEAR": float(amt_req_credit_bureau_year) if amt_req_credit_bureau_year is not None else None
        }

        response = requests.post(API_URL, json=payload, timeout=10)

        if response.status_code == 200:
            result = response.json()
            decision = result["decision"]
            proba = result["probabilite"]
            latency = result["execution_time_ms"]

            color = "#2e7d32" if decision == "ACCORDÉ" else "#c62828"
            html_output = f"""
            <div style="padding: 20px; border-radius: 10px; background-color: {color}; color: white; text-align: center;">
                <h2 style="margin:0; font-size: 28px;">DÉCISION : {decision}</h2>
                <h3 style="margin:10px 0 0 0;">Probabilité de Défaut : {proba * 100:.2f}%</h3>
                <p style="margin:5px 0 0 0; font-size: 12px;">Temps de réponse API : {latency} ms</p>
            </div>
            """
            return html_output, result
        else:
            error_detail = response.json().get("detail", response.text)
            err_html = f"""
            <div style="padding: 15px; border-radius: 8px; background-color: #d32f2f; color: white;">
                <b>Erreur API ({response.status_code}) :</b> {error_detail}
            </div>
            """
            return err_html, {"error": error_detail}

    except requests.exceptions.RequestException as e:
        err_html = f"""
        <div style="padding: 15px; border-radius: 8px; background-color: #d32f2f; color: white;">
            <b>Erreur de Connexion :</b> Impossible de joindre l'API à {API_URL}.
        </div>
        """
        return err_html, {"error": str(e)}


with gr.Blocks(title="Scoring Crédit - Dashboard Décisionnel") as demo:

    gr.Markdown(
        """
        # Système d'Aide à la Décision - Scoring Crédit
        Sélectionnez un **ID Client** chargé en mémoire au hasard ou adaptez ses données. 
        Toutes les variables requises par le schéma Pydantic sont configurées ci-dessous.
        """
    )

    with gr.Row():
        with gr.Column(scale=2):
            with gr.Tab("1. Identité & Démographie"):
                sk_id_curr = gr.Dropdown(choices=CLIENT_IDS, value=CLIENT_IDS[0] if CLIENT_IDS else "", allow_custom_value=True, label="ID Client (SK_ID_CURR) ou saisir un nouveau")
                code_gender = gr.Radio(choices=["F", "M"], value="F", label="CODE_GENDER - Genre")
                age_years = gr.Slider(minimum=18, maximum=100, value=35, step=1, label="Âge (Converti depuis DAYS_BIRTH)")
                cnt_children = gr.Number(label="CNT_CHILDREN - Nombre d'enfants", value=0, precision=0)
                cnt_fam_members = gr.Number(label="CNT_FAM_MEMBERS - Membres de la famille", value=1.0)
                name_family_status = gr.Dropdown(
                    choices=["Married", "Single / not married", "Civil marriage", "Separated", "Widow", "Unknown"],
                    value="Married", allow_custom_value=True, label="NAME_FAMILY_STATUS - Statut Familial"
                )
                name_education_type = gr.Dropdown(
                    choices=["Secondary / secondary special", "Higher education", "Incomplete higher", "Lower secondary", "Academic degree"],
                    value="Secondary / secondary special", allow_custom_value=True, label="NAME_EDUCATION_TYPE - Éducation"
                )
                days_registration = gr.Number(label="DAYS_REGISTRATION (Avis de domicile - années écoulées)", value=3.0)
                days_id_publish = gr.Number(label="DAYS_ID_PUBLISH (Pièce d'identité - années écoulées)", value=3.0)

            with gr.Tab("2. Prêt & Finances"):
                name_contract_type = gr.Radio(choices=["Cash loans", "Revolving loans"], value="Cash loans", label="NAME_CONTRACT_TYPE - Type de Prêt")
                amt_income_total = gr.Number(label="AMT_INCOME_TOTAL - Revenu Annuel ($)", value=150000.0)
                amt_credit = gr.Number(label="AMT_CREDIT - Montant du Crédit ($)", value=450000.0)
                amt_annuity = gr.Number(label="AMT_ANNUITY - Annuité du Prêt ($)", value=22500.0)
                amt_goods_price = gr.Number(label="AMT_GOODS_PRICE - Prix du bien financé ($)", value=450000.0)

            with gr.Tab("3. Emploi & Logement"):
                years_employed = gr.Number(label="Ancienneté Emploi (Converti depuis DAYS_EMPLOYED)", value=5.0)
                name_income_type = gr.Dropdown(
                    choices=["Working", "Commercial associate", "Pensioner", "State servant", "Unemployed", "Student", "XNA"],
                    value="Working", allow_custom_value=True, label="NAME_INCOME_TYPE - Source de revenu"
                )
                occupation_type = gr.Dropdown(
                    choices=["Laborers", "Sales staff", "Core staff", "Managers", "Drivers", "High skill tech staff", "Accountants", "Medicine staff", "Security staff", "Cleaning staff", "Private service staff", "Cooking staff", "Low-skill Laborers", "HR staff", "IT staff", "XNA"],
                    value="Laborers", allow_custom_value=True, label="OCCUPATION_TYPE - Profession"
                )
                organization_type = gr.Dropdown(
                    choices=["Business Entity Type 3", "Self-employed", "Other", "Medicine", "Government", "School", "Trade: type 7", "Kindergarten", "XNA"],
                    value="Business Entity Type 3", allow_custom_value=True, label="ORGANIZATION_TYPE - Organisation"
                )
                name_housing_type = gr.Dropdown(
                    choices=["House / apartment", "With parents", "Rented apartment", "Municipal apartment", "Office apartment", "Co-op apartment"],
                    value="House / apartment", allow_custom_value=True, label="NAME_HOUSING_TYPE - Type de logement"
                )
                flag_own_car = gr.Radio(choices=["Y", "N"], value="N", label="FLAG_OWN_CAR - Véhicule")
                flag_own_realty = gr.Radio(choices=["Y", "N"], value="Y", label="FLAG_OWN_REALTY - Immeuble")

            with gr.Tab("4. Région & Contact"):
                region_rating_client = gr.Radio(choices=[1, 2, 3], value=2, label="REGION_RATING_CLIENT - Note Région")
                reg_city_not_live_city = gr.Radio(choices=[0, 1], value=0, label="REG_CITY_NOT_LIVE_CITY (1 si adresse contact != résid.)")
                reg_city_not_work_city = gr.Radio(choices=[0, 1], value=0, label="REG_CITY_NOT_WORK_CITY (1 si adresse travail != résid.)")
                days_last_phone_change = gr.Number(label="DAYS_LAST_PHONE_CHANGE (années depuis chang. tel)", value=1.0)
                
                with gr.Row():
                    flag_mobil = gr.Radio(choices=[0, 1], value=1, label="FLAG_MOBIL")
                    flag_emp_phone = gr.Radio(choices=[0, 1], value=1, label="FLAG_EMP_PHONE")
                    flag_work_phone = gr.Radio(choices=[0, 1], value=0, label="FLAG_WORK_PHONE")
                with gr.Row():
                    flag_cont_mobile = gr.Radio(choices=[0, 1], value=1, label="FLAG_CONT_MOBILE")
                    flag_phone = gr.Radio(choices=[0, 1], value=0, label="FLAG_PHONE")
                    flag_email = gr.Radio(choices=[0, 1], value=0, label="FLAG_EMAIL")

            with gr.Tab("5. Scores Ext. & Documents"):
                ext_source_1 = gr.Slider(minimum=0.0, maximum=1.0, value=0.5, label="EXT_SOURCE_1")
                ext_source_2 = gr.Slider(minimum=0.0, maximum=1.0, value=0.5, label="EXT_SOURCE_2")
                ext_source_3 = gr.Slider(minimum=0.0, maximum=1.0, value=0.5, label="EXT_SOURCE_3")
                
                gr.Markdown("#### Documents Fournis (Flags)")
                with gr.Row():
                    flag_document_2 = gr.Radio(choices=[0, 1], value=0, label="FLAG_DOCUMENT_2")
                    flag_document_3 = gr.Radio(choices=[0, 1], value=1, label="FLAG_DOCUMENT_3")
                    flag_document_4 = gr.Radio(choices=[0, 1], value=0, label="FLAG_DOCUMENT_4")
                    flag_document_5 = gr.Radio(choices=[0, 1], value=0, label="FLAG_DOCUMENT_5")

            with gr.Tab("6. Social & Bureau Crédit"):
                gr.Markdown("#### Entourage Social (Social Circle)")
                with gr.Row():
                    obs_30_cnt_social_circle = gr.Number(label="OBS_30_CNT_SOCIAL_CIRCLE", value=0.0)
                    def_30_cnt_social_circle = gr.Number(label="DEF_30_CNT_SOCIAL_CIRCLE", value=0.0)
                with gr.Row():
                    obs_60_cnt_social_circle = gr.Number(label="OBS_60_CNT_SOCIAL_CIRCLE", value=0.0)
                    def_60_cnt_social_circle = gr.Number(label="DEF_60_CNT_SOCIAL_CIRCLE", value=0.0)

                gr.Markdown("#### Demandes Bureau Crédit (AMT_REQ_CREDIT_BUREAU_*)")
                with gr.Row():
                    amt_req_credit_bureau_hour = gr.Number(label="..._HOUR", value=0.0)
                    amt_req_credit_bureau_day = gr.Number(label="..._DAY", value=0.0)
                    amt_req_credit_bureau_week = gr.Number(label="..._WEEK", value=0.0)
                with gr.Row():
                    amt_req_credit_bureau_mon = gr.Number(label="..._MON", value=0.0)
                    amt_req_credit_bureau_qrt = gr.Number(label="..._QRT", value=0.0)
                    amt_req_credit_bureau_year = gr.Number(label="..._YEAR", value=0.0)

            submit_btn = gr.Button("Lancer l'Analyse du Risque", variant="primary")

        # --- Colonne Droite : Résultats ---
        with gr.Column(scale=1):
            gr.Markdown("### Résultat du Scoring")
            result_html = gr.HTML(value="<p style='text-align:center;'>Sélectionnez un client et cliquez sur Lancer.</p>")
            result_json = gr.JSON(label="Détail Réponse API")

    # --- Liste ordonnée de tous les composants de l'interface ---
    all_form_inputs = [
        name_contract_type, code_gender, flag_own_car, flag_own_realty,
        cnt_children, age_years, years_employed, days_registration, days_id_publish,
        name_income_type, name_education_type, name_family_status, name_housing_type,
        occupation_type, organization_type, cnt_fam_members, region_rating_client,
        reg_city_not_live_city, reg_city_not_work_city, amt_income_total, amt_credit,
        amt_annuity, amt_goods_price, ext_source_1, ext_source_2, ext_source_3,
        flag_mobil, flag_emp_phone, flag_work_phone, flag_cont_mobile, flag_phone,
        flag_email, days_last_phone_change, flag_document_2, flag_document_3,
        flag_document_4, flag_document_5, obs_30_cnt_social_circle, def_30_cnt_social_circle,
        obs_60_cnt_social_circle, def_60_cnt_social_circle, amt_req_credit_bureau_hour,
        amt_req_credit_bureau_day, amt_req_credit_bureau_week, amt_req_credit_bureau_mon,
        amt_req_credit_bureau_qrt, amt_req_credit_bureau_year
    ]

    # --- Connexion des événements ---
    
    # Auto-remplissage au changement du menu déroulant SK_ID_CURR
    sk_id_curr.change(
        fn=populate_form,
        inputs=[sk_id_curr],
        outputs=all_form_inputs
    )

    # Soumission du formulaire
    submit_btn.click(
        fn=send_prediction_request,
        inputs=[sk_id_curr] + all_form_inputs,
        outputs=[result_html, result_json]
    )

if __name__ == "__main__":
    demo.launch(
        server_name=os.getenv("GRADIO_SERVER_NAME", "0.0.0.0"),
        server_port=int(os.getenv("GRADIO_SERVER_PORT", "7860")),
        theme=gr.themes.Soft()
    )