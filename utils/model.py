from datetime import datetime, timedelta
import numpy as np
import pandas as pd
from utils.data import df
from scipy.integrate import odeint
from scipy.optimize import minimize
from statsmodels.tsa.arima.model import ARIMA

class SIR:
    def __init__(self, population, n):
        self.beta = []
        self.gamma = []
        self.mu = []
        self.population = population
        self.n = n
        
    def obj_func(self, params, data, t, y0):
        beta, gamma, mu = params
        preds = odeint(self.model, y0, t, args=(beta, gamma, mu))
        err = np.sum(np.square(preds.T - data))
        
        return err
        
    def fit(self, y0, t, data):
        results = np.array([[x] for x in y0])
        for i in range(0, len(t), self.n):
            ti = list(range(len(t[i:i + self.n])))
            
            # Find parameter values that minimizes the error between the model and the real data
            params = minimize(self.obj_func, (0, 0, 0), args=(data[:, i:i + self.n], ti, results[:, -1]), bounds=((0, None), (0, None), (0, None)))
            beta, gamma, mu = params.x
            
            self.beta.append(beta)
            self.gamma.append(gamma)
            self.mu.append(mu)

            result_i = odeint(self.model, results[:, -1], ti, args=(beta, gamma, mu))
            results = np.concatenate((results, result_i.T), axis=1)
            
            self.last_value = results[:, -1]
            
        # self.beta = pd.Series(self.beta).ewm(alpha=0.2).mean().values
        # self.gamma = pd.Series(self.gamma).ewm(alpha=0.2).mean().values
        # self.mu = pd.Series(self.mu).ewm(alpha=0.2).mean().values
        self.results = results
    
    def predict(self, days):
        weights = [0.5, 0.3, 0.2]
        beta = np.average(self.beta[-3:], weights=weights)
        gamma = np.average(self.gamma[-3:], weights=weights)
        mu = np.average(self.mu[-3:], weights=weights)
        
        t = list(range(days))
        
        return odeint(self.model, self.last_value, t, args=(beta, gamma, mu)).T
    
    def model(self, y, t, beta, gamma, mu):
        Y = np.zeros((4))
        V = y
        #dS/dt SUCEPTIBLES
        Y[0] = - beta * V[0] * V[1] / self.population
        #dI/dt INFECTADOS
        Y[1] = beta * V[0] * V[1] / self.population - gamma * V[1] - mu * V[1]
        #dR/dt RECUPERADOS
        Y[2] = gamma * V[1] #gama1-20*infectados1-20+gama20-40*infectados20-40
        #dM/dt MUERTES
        Y[3] = mu * V[1]

        return Y

poblacion_ciudades = pd.read_csv("spreadsheets/top20ciudades.csv")
dates_df = pd.DataFrame(df["fecha_reporte_web"].unique(), columns=["fecha_reporte_web"])

# 5 ciudades con mas casos reales
cities = df.groupby(by=['ciudad_de_ubicaci_n'], as_index=False).count()[['ciudad_de_ubicaci_n', 'edad']]
cities.rename(columns={"edad": "casos"}, inplace=True)
top_5 = cities.sort_values("casos", ascending=False)["ciudad_de_ubicaci_n"].tolist()[:5]

# Número de días para hacer el pronostico
now = datetime.now()
days = 60
dates = [(now + timedelta(days=i)).strftime("%Y-%m-%d") for i in range(days)]

cities_data = dict()
preds = {
    "dates": dates,
    "cities": []
}

for city in top_5:
    cities_data[city] = dict()
    city_pop = list(poblacion_ciudades[poblacion_ciudades['Ciudad']==city]['Población'])[0]
    df_city = df[df["ciudad_de_ubicaci_n"] == city]
    
    history_cases = df_city.groupby("fecha_reporte_web", as_index=False).count()
    history_recovery = df_city.groupby("fecha_recuperado", as_index=False).count()
    history_death = df_city.groupby("fecha_de_muerte", as_index=False).count()
    history_cases = pd.merge(dates_df, history_cases, on="fecha_reporte_web", how="left", sort=True).fillna(0)["sexo"].values
    history_recovery = pd.merge(dates_df.rename(columns={"fecha_reporte_web": "fecha_recuperado"}), history_recovery, on="fecha_recuperado", how="left", sort=True).fillna(0)["sexo"].values
    history_death = pd.merge(dates_df.rename(columns={"fecha_reporte_web": "fecha_de_muerte"}), history_death, on="fecha_de_muerte", how="left", sort=True).fillna(0)["sexo"].values
    
    # Derivadas
    dSdt = - history_cases
    dRdt = history_recovery
    dMdt = history_death
    dIdt = - dSdt - dRdt - dMdt
    
    # Acumulados
    S = city_pop + dSdt.cumsum()
    R = dRdt.cumsum()
    M = dMdt.cumsum()
    I = dIdt.cumsum()

    data = np.array([S, I, R, M])

    sir_model = SIR(city_pop, 3)

    y0 = [city_pop, 1, 0, 0]
    t = list(range(data.shape[1]))

    sir_model.fit(y0, t, data)
    arima_model = [ARIMA(d, order=(1,0,0), enforce_stationarity=False).fit() for d in data]

    sir_pred = sir_model.predict(days)
    arima_pred = np.array([m.forecast(steps=days) for m in arima_model])

    sigmoid = lambda x: 1 / (1 + np.exp(-x))
    sir_weights = np.hstack((np.linspace(0, 0.8, 5), np.linspace(0.8, 1, 10, endpoint=True), np.ones(45)))
    arima_weights = 1 - sir_weights

    city_pred = sir_weights * sir_pred + arima_weights * arima_pred

    preds["cities"].append({
        "Ciudad": city,
        "Susceptibles": city_pred[0].tolist(),
        "Infectados": city_pred[1].tolist(),
        "Recuperados": city_pred[2].tolist(),
        "Muertos": city_pred[3].tolist(),

    }) 