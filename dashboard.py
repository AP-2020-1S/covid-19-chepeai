import numpy as np
import pandas as pd
from data import df

dates = pd.DataFrame(df["fecha_reporte_web"].unique(), columns=["fecha_reporte_web"])

def format_data(df):
    df_copy = df.copy()
    data = dict()

    active_cases_per_city = df_copy[df_copy["fecha_recuperado"].isnull() & df_copy["fecha_de_muerte"].isnull()]["ciudad_de_ubicaci_n"].value_counts()[:5].to_dict()
    data["summary"] = {
        "total": len(df_copy),
        "recovered": len(df_copy.dropna(subset=["fecha_recuperado"])),
        "deaths": len(df_copy.dropna(subset=["fecha_de_muerte"])),
        "sex": df_copy["sexo"].value_counts().to_dict(),
        "city": active_cases_per_city
    }

    history = df_copy.groupby("fecha_reporte_web", as_index=False).count()
    history = pd.merge(dates, history, on="fecha_reporte_web", how="left", sort=True).fillna(0)
    history["fecha_reporte_web"] = history["fecha_reporte_web"].apply(lambda x: x[:10])
    data["history"] = {
        "date": history["fecha_reporte_web"].tolist(),
        "total": history["sexo"].tolist(),
        "recovered": history["fecha_recuperado"].tolist(),
        "deaths": history["fecha_de_muerte"].tolist()
    }

    return data


data = format_data(df.copy())
data_by_province = {province: format_data(df[df["departamento"] == province]) for province in df["departamento"].unique()}