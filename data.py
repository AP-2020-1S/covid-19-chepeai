"""
Archivo para descargar los datos de https://www.datos.gov.co para ser
usados por los modulos para crear los dashboard

"""
import unicodedata
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

socrata_domain = "www.datos.gov.co"
socrata_dataset_identifier = "gt2j-8ykr"

client = Socrata(socrata_domain, None)
results = client.get(socrata_dataset_identifier,limit=1000000000)
df = pd.DataFrame.from_dict(results)
# df.to_csv("data.csv")
# df = pd.read_csv("data.csv")
df["sexo"] = df["sexo"].apply(lambda x: "Masculino" if x.upper() == "M" else "Femenino")
df["departamento"] = df["departamento"].apply(lambda x: province_name_map.get(x, replace_tildes(x)).upper())