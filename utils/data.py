"""
Archivo para descargar los datos de https://www.datos.gov.co para ser
usados por los modulos para crear los dashboard

"""
import os
import unicodedata
import numpy as np
import pandas as pd
from sodapy import Socrata

def replace_tildes(x):
    normalized_x = unicodedata.normalize('NFKD', x)
    normalized_x = normalized_x.encode('ASCII', 'ignore').decode('utf-8')
    normalized_x = normalized_x.upper()

    return normalized_x

province_name_map = {
    "Bogotá D.C.": "SANTAFE DE BOGOTA D.C",
    "Barranquilla D.E.": "ATLANTICO",
    "Cartagena D.T. y C.": "BOLIVAR",
    "Santa Marta D.T. y C.": "MAGDALENA",
    "Buenaventura D.E.": "VALLE DEL CAUCA",
    "Nariño": "NARIÑO"
}

#Descarga de la información
socrata_domain = "www.datos.gov.co"
client = Socrata(socrata_domain, None)

if os.path.exists("./spreadsheets/data_covid.csv"):
    df = pd.read_csv('./spreadsheets/data_covid.csv')
else:
    socrata_dataset_identifier = "gt2j-8ykr"
    results = client.get(socrata_dataset_identifier,limit=1000000000)

    df = pd.DataFrame.from_dict(results)
    df.to_csv("./spreadsheets/data_covid.csv", index=False)

if os.path.exists("./spreadsheets/pruebas_covid.csv"):
    df_pruebas = pd.read_csv('./spreadsheets/pruebas_covid.csv')
else:
    socrata_dataset_identifier = "8835-5baf"
    results = client.get(socrata_dataset_identifier,limit=1000000000)

    df_pruebas = pd.DataFrame.from_dict(results)
    df_pruebas.to_csv('./spreadsheets/pruebas_covid.csv', index=False)
    
    


#Estandarización
df["sexo"] = df["sexo"].apply(lambda x: "Masculino" if x.upper() == "M" else "Femenino")
df["departamento"] = df["departamento"].apply(lambda x: province_name_map.get(x, replace_tildes(x)).upper())

df.fillna(value=np.NaN, inplace=True)

to_drop = ['id_de_caso', 'fis', 'c_digo_divipola', 'codigo_departamento','codigo_pais']

df.drop(labels=to_drop, axis=1, inplace=True)

#date_columns = ['fecha_de_muerte',
#                'fecha_recuperado',
#                'fecha_reporte_web',
#                'fecha_diagnostico',
#                'fecha_de_notificaci_n']

#for col in date_columns:
 #   df[col] = pd.to_datetime(df[col])

df['edad'] = df['edad'].apply(lambda x: int(x))

cat_cols = ['sexo', 'tipo', 'estado', 'atenci_n', 'pertenencia_etnica', 'tipo_recuperaci_n']

for col in cat_cols:
    df[col] = df[col].apply(lambda x: str(x).strip().capitalize() if str(x) != 'N/A' else str(x).strip())

#Estandarización data pruebas
df_pruebas.replace("Acumulado Feb", value="2020-02-29T00:00:00.000", inplace = True)
df_pruebas['fecha'] = pd.to_datetime(df_pruebas['fecha'])
df_pruebas['acumuladas']= df_pruebas['acumuladas'].astype('int')
df_pruebas['acumuladas'] = df_pruebas['acumuladas'].diff()

df_pruebas.fillna(0, inplace=True)