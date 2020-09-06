from datetime import datetime, timedelta
import numpy as np

## Insertar entrenamiento del model aca y transformar la data al formato que se muestra al final

# Remplazar por las 5 ciudades con mas casos reales
cities = ["Bogota", "Medellin", "Cali", "Barranquilla", "Cartagena"]
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
        } for city in cities
    ]
}