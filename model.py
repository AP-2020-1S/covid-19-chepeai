from datetime import datetime, timedelta
import numpy as np
from data import df

## Insertar entrenamiento del model aca y transformar la data al formato que se muestra al final

# Remplazar por las 5 ciudades con mas casos reales
cities = df.groupby(by=['ciudad_de_ubicaci_n'], as_index=False).count()[['ciudad_de_ubicaci_n', 'edad']]
cities.rename(columns={"edad": "casos"}, inplace=True)
top_cities = cities.sort_values("casos", ascending=False)["ciudad_de_ubicaci_n"].tolist()[:5]

now = datetime.now()
# Reemplazar con el rango de fecha real que se va a considerar
dates = [(now + timedelta(days=i)).strftime("%Y-%m-%d") for i in range(91)]
# Borrar esta linea cuando se tenga la data real
mock_data = lambda : (np.linspace(1, 2, 90) + (2 * np.random.rand(90) - 1)).tolist()

# Este es el formato en que se debe enviar la data para ser pintada en el reporte
preds = {
    "dates": dates, 
    "cities": [
        {
            "Ciudad": city,
            "Susceptibles": mock_data(),
            "Infectados": mock_data(),
            "Recuperados": mock_data(),
            "Muertos": mock_data(),
        } for city in top_cities
    ]
}