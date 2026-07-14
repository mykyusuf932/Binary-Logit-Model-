# -*- coding: utf-8 -*-

"""
================================================================================
EV TERCIHI - MODIFIED BINARY LOGIT MODELI
Biogeme ile secilmis degisken seti uzerinden tahmin

Model ozeti:
- Bagimli degisken: CHOSE_LONG_RANGE_EV
- Tercih kodlamasi:
    1 = kisa menzilli EV
    2 = uzun menzilli EV
- Model, uzun menzilli EV secimini aciklamak icin secilmis ekonomik,
  demografik ve davranissal degiskenleri kullanir.
- Fiyat ve menzil degiskenleri uzerinden WTP hesaplanir.
================================================================================
"""

import os
import sys
import numpy as np
import pandas as pd

from scipy.stats import norm

import biogeme.database as db
import biogeme.biogeme as bio
from biogeme import models
from biogeme.expressions import Beta, Variable

try:
    import biogeme.version as ver
    BIOGEME_VERSION = ver.getVersion()
except Exception:
    BIOGEME_VERSION = "Unknown"


# ============================================================
# 1. DOSYA AYARLARI
# ============================================================

WORK_DIR = r"D:\Drive\OneDrive - Gebze Teknik Üniversitesi\Tez\EV_Biogeme_Work"

INPUT_FILE = "EV_ALL_v6_tüik.xlsx"
INPUT_SHEET = "Model_Data"

OUTPUT_FILE = "Biogeme_Modified_Binary_Logit_Results_Final.xlsx"

# True  = modified model, full specification ile ayni orneklem uzerinden tahmin edilir.
# False = orneklem yalnizca modified model degiskenlerine gore belirlenir.
USE_FULL_SPECIFICATION_SAMPLE = True

input_path = os.path.join(WORK_DIR, INPUT_FILE)
output_path = os.path.join(WORK_DIR, OUTPUT_FILE)

if not os.path.exists(input_path):
    print("Excel dosyasi bulunamadi:")
    print(input_path)
    print("\nKontrol et:")
    print("1) Excel dosyasi WORK_DIR klasorunde mi?")
    print("2) Dosya adi tam olarak EV_ALL_v6_tüik.xlsx mi?")
    sys.exit()


# ============================================================
# 2. VERIYI OKU
# ============================================================

df = pd.read_excel(input_path, sheet_name=INPUT_SHEET)

print("Veri okundu.")
print("Satir sayisi:", len(df))
print("Sutun sayisi:", len(df.columns))
print("Biogeme surumu:", BIOGEME_VERSION)


# ============================================================
# 3. BAGIMLI DEGISKEN
# ============================================================

choice_col = "CHOSE_LONG_RANGE_EV"

if choice_col not in df.columns:
    raise ValueError(f"{choice_col} sutunu bulunamadi.")

df[choice_col] = pd.to_numeric(df[choice_col], errors="coerce")

print("\nCHOSE_LONG_RANGE_EV ham dagilimi:")
print(df[choice_col].value_counts(dropna=False).sort_index())

# Biogeme tercih kodu:
# 1 = kisa menzilli EV
# 2 = uzun menzilli EV

df["CHOICE_BIOGEME"] = np.nan
df.loc[df[choice_col] == 0, "CHOICE_BIOGEME"] = 1
df.loc[df[choice_col] == 1, "CHOICE_BIOGEME"] = 2


# ============================================================
# 4. YAS VE GELIR DEGISKENLERI
# ============================================================

AGE_VARIABLES = [
    "AGE_18_28",
    "AGE_29_41",
    "AGE_57_PLUS",
]

INCOME_VARIABLE = "PROXY_HIGH_INCOME"

required_demographic_columns = AGE_VARIABLES + [INCOME_VARIABLE]
missing_demographic_columns = [
    c for c in required_demographic_columns if c not in df.columns
]

if missing_demographic_columns:
    print("\nAsagidaki yas/gelir degiskenleri Excel dosyasinda bulunamadi:")
    for c in missing_demographic_columns:
        print("-", c)
    sys.exit()

for col in required_demographic_columns:
    df[col] = pd.to_numeric(df[col], errors="coerce")

print("\nYas dummy dagilimlari:")
for col in AGE_VARIABLES:
    print("\n", col, sep="")
    print(df[col].value_counts(dropna=False).sort_index())

print("\nPROXY_HIGH_INCOME dagilimi:")
print(df[INCOME_VARIABLE].value_counts(dropna=False).sort_index())


# ============================================================
# 5. FULL SPECIFICATION DEGISKEN SETI
#    Bu liste, orneklem uyumunu korumak icin kullanilir.
# ============================================================

FULL_SPECIFICATION_VARIABLES = [
    # Secim tasarimi / alternatif farklari
    "PRICE_DIFF_M_TL",
    "RANGE_DIFF_100KM",

    # Yas dummy degiskenleri
    "AGE_18_28",
    "AGE_29_41",
    "AGE_57_PLUS",

    # Demografik degiskenler
    "FEMALE",
    "MARRIED",
    "KID_NUMBER",
    "UNIVERSITY_GRAD_OR_ABOVE",
    "PROXY_HIGH_INCOME",
    "EMPLOYED",

    # Ulasim ve arac sahipligi
    "VEHICLE_AVAILABLE",
    "LICENCE_EXPERIENCED",
    "PRIVATE_TRANSPORT",

    # Gelecek satin alma ve EV gecmisi
    "PLAN_BUY_2Y",
    "PAST_EV_PURCHASE",

    # Davranissal profil degiskenleri
    "ENVIRONMENTALIST",
    "TECH_SEEKER",
    "PERFORMANCE_SEEKER",
    "INCENTIVE_SEEKER",
    "RANGE_SENSITIVITY",
    "WILLINGNESS_TO_PAY_LONG_RANGE",
    "CHARGE_STATION_WORRY",
    "CHARGING_ACCESS_SEEKER",
    "CHARGING_TIME_SENSITIVE",
    "RESALE_VALUE_SENSITIVE",
    "RELIABILITY_SEEKER",
    "PRICE_SENSITIVE",
]


# ============================================================
# 6. MODIFIED MODELDE KULLANILACAK DEGISKENLER
# ============================================================

MODEL_VARIABLES_BASE = [
    # Secim tasarimi / alternatif farklari
    "PRICE_DIFF_M_TL",
    "RANGE_DIFF_100KM",

    # Demografik degiskenler
    "PROXY_HIGH_INCOME",
    "EMPLOYED",

    # Ulasim ve arac sahipligi
    "VEHICLE_AVAILABLE",

    # Davranissal profil degiskenleri
    "RANGE_SENSITIVITY",
    "WILLINGNESS_TO_PAY_LONG_RANGE",
    "CHARGING_ACCESS_SEEKER",
    "RELIABILITY_SEEKER",
    "PRICE_SENSITIVE",
]


# ============================================================
# 7. MODIFIED MODELDE KULLANILMAYAN DEGISKEN GRUPLARI
# ============================================================

VARIABLES_NOT_IN_MODEL = [
    # Yuksek eksik veri veya modelleme kapsami disinda birakilanlar
    "CHARGING_PLACE",
    "ELECTRIC_CAR_AGAIN",
    "ELECTRIC_CAR_MAINTENANCE_FEE",
    "LOW_MAINT_ENERGY_COST_ATTITUDE",
    "CAR_AGE",
    "CAR_USAGE_HIGH",
    "CAR_CHANGE_SOON",
    "FUEL_HYBRID",
    "FUEL_EV_OR_PHEV",
    "FUEL_FOSSIL",
    "Q37_LOW_MAINTENANCE_COST",
    "DISTANCE_NEAR",

    # Ham Q37 maddeleri yerine davranissal profil degiskenleri kullanilir
    "Q37_GOV_INCENTIVES",
    "Q37_BATTERY_RANGE",
    "Q37_CHARGING_TIME",
    "Q37_CHARGING_STATION_AVAILABILITY",
    "Q37_RESALE_VALUE",
    "Q37_RELIABILITY_DURABILITY",
    "Q37_PURCHASE_PRICE",

    # WTP icin modelde WILLINGNESS_TO_PAY_LONG_RANGE kullanilir
    "WTP_LONG_RANGE",

    # Ham tutumsal Likert maddeleri yerine davranissal profil degiskenleri kullanilir
    "CLIMATE_CHANGE_CONCERN",
    "EV_ENV_HEALTH_BENEFIT",
    "EV_TECH_ADEQUACY",
    "CHARGING_INFRA_CONCERN",
    "RANGE_ANXIETY",
    "SECOND_HAND_VALUE_RISK",
    "GOV_INCENTIVES_EFFECTIVENESS",
    "EMISSION_REDUCTION_INTENTION",
    "ACCELERATION_PERFORMANCE_IMPORTANCE",
    "EV_MOTOR_DURABILITY_PERCEPTION",
    "RANGE_IMPORTANCE_PURCHASE",
    "FINANCIAL_INCENTIVES_IMPORTANCE",

    # Modelde kullanilmayan yas ve gelir degiskenleri
    "AGE",
    "AGE_CENTERED",
    "AGE_GROUP",
    "AGE_GROUP_NEW",
    "AGE_G1",
    "AGE_G3",
    "INCOME_HIGH",
]

FULL_SPECIFICATION_NOT_SELECTED = [
    "AGE_18_28",
    "AGE_29_41",
    "AGE_57_PLUS",
    "FEMALE",
    "MARRIED",
    "KID_NUMBER",
    "UNIVERSITY_GRAD_OR_ABOVE",
    "LICENCE_EXPERIENCED",
    "PRIVATE_TRANSPORT",
    "PLAN_BUY_2Y",
    "PAST_EV_PURCHASE",
    "ENVIRONMENTALIST",
    "TECH_SEEKER",
    "PERFORMANCE_SEEKER",
    "INCENTIVE_SEEKER",
    "CHARGE_STATION_WORRY",
    "CHARGING_TIME_SENSITIVE",
    "RESALE_VALUE_SENSITIVE",
]

TEN_PERCENT_ONLY_VARIABLES = [
    "AGE_57_PLUS",
]


# ============================================================
# 8. DEGISKEN KONTROLU
# ============================================================

variables_to_check = sorted(
    set(FULL_SPECIFICATION_VARIABLES + MODEL_VARIABLES_BASE)
)

missing_vars = [v for v in variables_to_check if v not in df.columns]

if missing_vars:
    print("\nAsagidaki degiskenler Excel dosyasinda bulunamadi:")
    for v in missing_vars:
        print("-", v)
    sys.exit()

wrongly_used = [
    v for v in (VARIABLES_NOT_IN_MODEL + FULL_SPECIFICATION_NOT_SELECTED)
    if v in MODEL_VARIABLES_BASE
]

if wrongly_used:
    print("\nModel degisken listesi kontrol edilmeli:")
    for v in wrongly_used:
        print("-", v)
    sys.exit()

MODEL_VARIABLES = MODEL_VARIABLES_BASE.copy()

for col in ["CHOICE_BIOGEME"] + variables_to_check:
    df[col] = pd.to_numeric(df[col], errors="coerce")


# ============================================================
# 9. EKSIK VERI VE SABIT DEGISKEN KONTROLU
# ============================================================

AUTO_EXCLUDED_CONSTANT_VARIABLES = []

while True:
    if USE_FULL_SPECIFICATION_SAMPLE:
        sample_dropna_cols = ["CHOICE_BIOGEME"] + FULL_SPECIFICATION_VARIABLES
    else:
        sample_dropna_cols = ["CHOICE_BIOGEME"] + MODEL_VARIABLES

    missing_summary = df[sample_dropna_cols].isna().sum().reset_index()
    missing_summary.columns = ["Variable", "Missing_Count"]
    missing_summary = missing_summary.sort_values("Missing_Count", ascending=False)

    df_model = df[sample_dropna_cols].dropna().copy()

    if len(df_model) == 0:
        raise ValueError("Model icin uygun gozlem kalmadi. Degiskenleri ve eksik veriyi kontrol et.")

    nunique = df_model[MODEL_VARIABLES].nunique(dropna=True)
    constant_vars = nunique[nunique < 2].index.tolist()

    if not constant_vars:
        break

    print("\nTek degerli degiskenler modelden cikariliyor:")
    for v in constant_vars:
        print("-", v)

    AUTO_EXCLUDED_CONSTANT_VARIABLES.extend(constant_vars)
    MODEL_VARIABLES = [v for v in MODEL_VARIABLES if v not in constant_vars]


df_model_estimation = df_model[["CHOICE_BIOGEME"] + MODEL_VARIABLES].copy()

print("\nModified modelde kullanilan satir sayisi:", len(df_model_estimation))
print("Cikarilan satir sayisi:", len(df) - len(df_model_estimation))
print("Orneklem belirleme kurali:", "Full specification orneklemi" if USE_FULL_SPECIFICATION_SAMPLE else "Modified model degiskenleri")

print("\nBagimli degisken dagilimi, Biogeme kodu:")
print(df_model_estimation["CHOICE_BIOGEME"].value_counts().sort_index())

print("\nModified modelde kullanilan degisken sayisi:", len(MODEL_VARIABLES))

if AUTO_EXCLUDED_CONSTANT_VARIABLES:
    print("\nOtomatik cikarilan sabit degiskenler:")
    print(AUTO_EXCLUDED_CONSTANT_VARIABLES)
else:
    print("\nSabit degisken bulunmadi.")


# ============================================================
# 9B. NADIR / DENGESIZ DUMMY DEGISKEN KONTROLU
# ============================================================

RARE_THRESHOLD = 10
rare_dummy_records = []

for var in MODEL_VARIABLES:
    values = df_model_estimation[var].dropna().unique()
    values_set = set(values)

    if values_set.issubset({0, 1, 0.0, 1.0}):
        value_counts = df_model_estimation[var].value_counts()

        count_0 = int(value_counts.get(0, 0))
        count_1 = int(value_counts.get(1, 0))
        min_count = min(count_0, count_1)

        if min_count < RARE_THRESHOLD:
            rare_dummy_records.append({
                "Variable": var,
                "Count_0": count_0,
                "Count_1": count_1,
                "Minimum_Group_Count": min_count,
                "Warning": "Rare / imbalanced dummy variable; interpret coefficient carefully"
            })

rare_dummy_table = pd.DataFrame(
    rare_dummy_records,
    columns=[
        "Variable",
        "Count_0",
        "Count_1",
        "Minimum_Group_Count",
        "Warning"
    ]
)

if not rare_dummy_table.empty:
    print("\nCok dengesiz / az gozlemli dummy degiskenler bulundu.")
    print(rare_dummy_table[["Variable", "Count_0", "Count_1", "Minimum_Group_Count"]])
else:
    print("\nCok dengesiz dummy degisken bulunmadi.")


# ============================================================
# 10. BIOGEME DATABASE
# ============================================================

database = db.Database("EV_MODIFIED_BINARY_LOGIT_FINAL", df_model_estimation)


# ============================================================
# 11. FAYDA FONKSIYONU
# ============================================================

ASC_LONG = Beta("ASC_LONG", 0, None, None, 0)

V_LONG = ASC_LONG

for var in MODEL_VARIABLES:
    beta_name = "B_" + var
    beta = Beta(beta_name, 0, None, None, 0)
    V_LONG += beta * Variable(var)

V = {
    1: 0,
    2: V_LONG
}

AV = {
    1: 1,
    2: 1
}

logprob = models.loglogit(V, AV, Variable("CHOICE_BIOGEME"))


# ============================================================
# 12. MODELI TAHMIN ET
# ============================================================

biogeme = bio.BIOGEME(database, logprob)

try:
    biogeme.model_name = "MODIFIED_BINARY_LOGIT_FINAL"
except Exception:
    biogeme.modelName = "MODIFIED_BINARY_LOGIT_FINAL"

print("\nBiogeme modified modeli tahmin ediliyor...")
raw_results = biogeme.estimate()
print("Tahmin tamamlandi.")


# ============================================================
# 13. SONUCLARI AL
# ============================================================

def get_parameters_dataframe(raw_results):
    result_objects = [raw_results]

    try:
        from biogeme.results_processing.estimation_results import EstimationResults
        result_objects.insert(0, EstimationResults(raw_results))
    except Exception:
        pass

    method_names = [
        "get_estimated_parameters",
        "getEstimatedParameters",
        "get_pandas_estimated_parameters"
    ]

    last_error = None

    for obj in result_objects:
        for method_name in method_names:
            try:
                method = getattr(obj, method_name)
                params_candidate = method()
                if params_candidate is not None:
                    return pd.DataFrame(params_candidate)
            except Exception as e:
                last_error = e

    print("\nBiogeme sonuc tablosu okunamadi.")
    print("Mevcut sonuc nesnesi tipi:", type(raw_results))
    print("Son istisna:", last_error)
    raise AttributeError("Biogeme parameter table could not be extracted.")


params = get_parameters_dataframe(raw_results)
params = params.reset_index()

if "Name" in params.columns:
    params = params.rename(columns={"Name": "Variable"})
elif "Parameter" in params.columns:
    params = params.rename(columns={"Parameter": "Variable"})
elif "index" in params.columns:
    params = params.rename(columns={"index": "Variable"})
elif params.columns[0] != "Variable":
    params = params.rename(columns={params.columns[0]: "Variable"})


def find_col(dataframe, possible_names):
    cols_lower = {c.lower().strip(): c for c in dataframe.columns}

    for name in possible_names:
        key = name.lower().strip()
        if key in cols_lower:
            return cols_lower[key]

    for c in dataframe.columns:
        c_low = c.lower()
        for name in possible_names:
            if name.lower() in c_low:
                return c

    return None


coef_col = find_col(params, ["Value", "Estimate", "Coefficient"])
rob_se_col = find_col(params, ["Rob. Std err", "Robust std err", "Robust SE", "Robust Std. Err."])
rob_p_col = find_col(params, ["Rob. p-value", "Robust p-value", "Robust p value"])

if coef_col is None:
    print("\nMevcut Biogeme sutunlari:")
    print(params.columns.tolist())
    raise ValueError("Katsayi sutunu bulunamadi.")

results_table = pd.DataFrame()
results_table["Variable"] = params["Variable"]
results_table["Coefficient"] = pd.to_numeric(params[coef_col], errors="coerce")

if rob_se_col is not None:
    results_table["Robust_SE"] = pd.to_numeric(params[rob_se_col], errors="coerce")
else:
    results_table["Robust_SE"] = np.nan

if rob_p_col is not None:
    results_table["Robust_p_value"] = pd.to_numeric(params[rob_p_col], errors="coerce")
else:
    results_table["Robust_p_value"] = np.nan

    if rob_se_col is not None:
        coef_numeric = pd.to_numeric(results_table["Coefficient"], errors="coerce")
        se_numeric = pd.to_numeric(results_table["Robust_SE"], errors="coerce")

        valid_se = (
            se_numeric.notna()
            & np.isfinite(se_numeric)
            & (se_numeric != 0)
        )

        z_value = pd.Series(np.nan, index=results_table.index)
        z_value.loc[valid_se] = coef_numeric.loc[valid_se] / se_numeric.loc[valid_se]

        p_value = pd.Series(np.nan, index=results_table.index)
        p_value.loc[valid_se] = 2 * (1 - norm.cdf(np.abs(z_value.loc[valid_se])))

        results_table["Robust_p_value"] = p_value

results_table["Significance_5pct"] = np.where(
    results_table["Robust_p_value"].isna(),
    "not available",
    np.where(
        results_table["Robust_p_value"] < 0.05,
        "significant",
        "insignificant"
    )
)

results_table["Significance_10pct"] = np.where(
    results_table["Robust_p_value"].isna(),
    "not available",
    np.where(
        results_table["Robust_p_value"] < 0.10,
        "significant at 10%",
        "not significant at 10%"
    )
)


# ============================================================
# 14. WTP HESABI
# ============================================================

def get_coef(parameter_name):
    row = results_table.loc[
        results_table["Variable"] == parameter_name,
        "Coefficient"
    ]
    if len(row) == 0:
        return np.nan
    return float(row.iloc[0])


beta_price = get_coef("B_PRICE_DIFF_M_TL")
beta_range = get_coef("B_RANGE_DIFF_100KM")

if pd.notna(beta_price) and beta_price != 0 and pd.notna(beta_range):
    wtp_million_tl_per_100km = -beta_range / beta_price
    wtp_tl_per_100km = wtp_million_tl_per_100km * 1_000_000
    wtp_tl_per_km = wtp_tl_per_100km / 100
else:
    wtp_million_tl_per_100km = np.nan
    wtp_tl_per_100km = np.nan
    wtp_tl_per_km = np.nan

wtp_table = pd.DataFrame({
    "Item": [
        "Beta price",
        "Beta range",
        "WTP million TL per 100 km",
        "WTP TL per 100 km",
        "WTP TL per km"
    ],
    "Value": [
        beta_price,
        beta_range,
        wtp_million_tl_per_100km,
        wtp_tl_per_100km,
        wtp_tl_per_km
    ]
})


# ============================================================
# 15. KULLANILAN / KULLANILMAYAN DEGISKEN LISTELERI
# ============================================================

used_variables_table = pd.DataFrame({
    "Used_Variable": MODEL_VARIABLES
})

not_in_model_table = pd.DataFrame({
    "Variable": VARIABLES_NOT_IN_MODEL,
    "Status": "Not included in the modified model specification"
})

not_selected_table = pd.DataFrame({
    "Variable": FULL_SPECIFICATION_NOT_SELECTED,
    "Status": "Not selected for the modified model specification"
})

auto_excluded_table = pd.DataFrame({
    "Variable": AUTO_EXCLUDED_CONSTANT_VARIABLES,
    "Status": "Automatically excluded: constant / single-value variable after sample selection"
})

excluded_variables_table = pd.concat(
    [not_in_model_table, not_selected_table, auto_excluded_table],
    ignore_index=True
)

ten_percent_table = pd.DataFrame({
    "Variable": TEN_PERCENT_ONLY_VARIABLES,
    "Status": "Below 10% threshold but above 5% threshold; not included in the modified model"
})

model_info = pd.DataFrame({
    "Item": [
        "Model type",
        "Model purpose",
        "Software",
        "Biogeme version",
        "Dependent variable",
        "Choice coding",
        "Variable selection rule",
        "Significance threshold used for variable selection",
        "Use full specification sample",
        "Sample rule",
        "Age variable treatment",
        "Age variables used in modified model",
        "Income variable used in model",
        "Income comparison group",
        "Income coefficient interpretation",
        "WTP variable used",
        "Number of observations in original data",
        "Number of observations used in model",
        "Number of excluded observations",
        "Number of variables in full specification",
        "Number of variables initially specified in modified model",
        "Number of variables finally used in modified model",
        "Automatically excluded constant variables",
        "Rare / imbalanced dummy variables",
        "10% only variables included?",
        "Hierarchical M1-M4 models",
        "Modified model",
        "Q41-Q46 p-value calculation",
        "Individual probability averages",
        "Market/real vehicle simulation"
    ],
    "Value": [
        "Binary logit",
        "Modified model using selected explanatory variables",
        "Biogeme",
        BIOGEME_VERSION,
        "CHOSE_LONG_RANGE_EV",
        "1 = short-range EV, 2 = long-range EV",
        "Variables were selected according to robust p-values in the full specification.",
        "5%",
        USE_FULL_SPECIFICATION_SAMPLE,
        "Full specification sample retained for comparability" if USE_FULL_SPECIFICATION_SAMPLE else "Sample based only on modified model variables",
        "No age dummy retained in the modified model",
        "None",
        "PROXY_HIGH_INCOME",
        "Non-high-income proxy group",
        "PROXY_HIGH_INCOME coefficient shows whether the high-income proxy group differs in long-range EV preference.",
        "WILLINGNESS_TO_PAY_LONG_RANGE",
        len(df),
        len(df_model_estimation),
        len(df) - len(df_model_estimation),
        len(FULL_SPECIFICATION_VARIABLES),
        len(MODEL_VARIABLES_BASE),
        len(MODEL_VARIABLES),
        ", ".join(AUTO_EXCLUDED_CONSTANT_VARIABLES) if AUTO_EXCLUDED_CONSTANT_VARIABLES else "None",
        ", ".join(rare_dummy_table["Variable"].tolist()) if not rare_dummy_table.empty else "None",
        "No",
        "No",
        "Yes",
        "No",
        "No",
        "No"
    ]
})


# ============================================================
# 16. EXCEL CIKTISI
# ============================================================

with pd.ExcelWriter(output_path, engine="openpyxl") as writer:
    results_table.to_excel(writer, sheet_name="Biogeme_Results", index=False)
    wtp_table.to_excel(writer, sheet_name="WTP", index=False)
    used_variables_table.to_excel(writer, sheet_name="Used_Variables", index=False)
    excluded_variables_table.to_excel(writer, sheet_name="Not_Used_Variables", index=False)
    ten_percent_table.to_excel(writer, sheet_name="10pct_Only_Not_Used", index=False)
    model_info.to_excel(writer, sheet_name="Model_Info", index=False)
    missing_summary.to_excel(writer, sheet_name="Missing_Data_Check", index=False)
    rare_dummy_table.to_excel(writer, sheet_name="Rare_Dummy_Check", index=False)

print("\nCikti dosyasi olusturuldu:")
print(output_path)

print("\nSonuc tablosu: Biogeme_Results")
print("WTP tablosu: WTP")
print("Kullanilan degisken listesi: Used_Variables")
print("Kullanilmayan degisken listesi: Not_Used_Variables")
print("10% duzeyinde kalan degiskenler: 10pct_Only_Not_Used")
print("Eksik veri kontrolu: Missing_Data_Check")
print("Nadir dummy kontrolu: Rare_Dummy_Check")
