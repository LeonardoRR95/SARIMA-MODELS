import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import statsmodels.api as sm
from statsmodels.tsa.stattools import adfuller
from statsmodels.tsa.stattools import kpss
from arch.unitroot import PhillipsPerron
from arch import arch_model
from scipy import stats 
import matplotlib.pyplot as plt
import statsmodels.api as sm
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf

# =============================================================================
#  DIRECTORIO DE TRABAJO
# =============================================================================

directorio = '/Users/leonardorebollarruelas/Documents/Modelo ARIMA'
os.chdir(directorio)

data = pd.read_csv('inflacion.csv')

data['Fecha'] = pd.date_range(start= '1990-01-01', end= '2024-12-01', freq = 'ME')

# Gráfico Serie Completa
sns.lineplot(x='Fecha', y='Inflacion', data=data)
plt.title('Inflacion')
plt.show()

# =============================================================================
#  DEFINICIÓN DE TIME SPAM
# =============================================================================

# Inflación Despues de 2008 

data8 = data[data['Fecha'] > "2008-01-01"]
data8['Inflacion_m'] = data8['Inflacion'].mean()

# Gráfico Serie Contemporanea 
sns.lineplot(x='Fecha', y='Inflacion', data=data8)
sns.lineplot(x='Fecha', y ='Inflacion_m', data=data8, linestyle='--')
plt.title('Inflación')
plt.show()

# =============================================================================
#  PRUEBAS DE ESTACIONARIEDAD
#   - Dickey-Fuller
#   - Phillips-Perron
#   - KPSS
#
#  CONSIDERANDO TRANSFORMACION LOGARITMICA Y COX-BOX
# =============================================================================

# Pruebas de estasionariedad

# Dickey-Fuller
dfp = adfuller(data8['Inflacion'])
print('Dickey-Fuller p-value: %f' % dfp[1])

# Phillips-Perron
pp_test = PhillipsPerron(data8['Inflacion'])
print(pp_test)

# Kwiatkowski-Phillips-Schmidt-Shin
kpss_test = kpss(data8['Inflacion'])
print('KPSS p-value: %f' % kpss_test[1])


# Pruebas de estasionariedad: logaritmo natural
data8['lnInflacio'] = np.log(data8['Inflacion']) # Transformación logaritmica


# Dickey-Fuller
dfp = adfuller(data8['lnInflacio'])
print('Dickey-Fuller p-value: %f' % dfp[1])

# Phillips-Perron
pp_test = PhillipsPerron(data8['lnInflacio'])
print(pp_test)

# Kwiatkowski-Phillips-Schmidt-Shin
kpss_test = kpss(data8['lnInflacio'])
print('KPSS p-value: %f' % kpss_test[1])

# Gráfico Serie Contemporanea 
sns.lineplot(x='Fecha', y='lnInflacio', data=data8)
plt.title('Inflación')
plt.show()

# Transformación Box - Box 

inflacion_box_cox, lambda_ = stats.boxcox(data8['Inflacion'])

data8['box_cox_Inflacion'] = inflacion_box_cox

# Pruebas de estasionariedad: Box - Cox

# Dickey-Fuller
dfp = adfuller(data8['box_cox_Inflacion'])
print('Dickey-Fuller p-value: %f' % dfp[1])

# Phillips-Perron
pp_test = PhillipsPerron(data8['box_cox_Inflacion'])
print(pp_test)

# Kwiatkowski-Phillips-Schmidt-Shin
kpss_test = kpss(data8['box_cox_Inflacion'])
print('KPSS p-value: %f' % kpss_test[1])

# Gráfico Serie Contemporanea 
sns.lineplot(x='Fecha', y='box_cox_Inflacion', data=data8)
plt.title('Inflación')
plt.show()


# =============================================================================
# DETERMINACIÓN DE MA Y AC MÁXIMOS PARA LA ESTIMACIÓN DE UN MODELO SARIMA
# =============================================================================

# Transformación en Logaritmos
# Función ACF
acf_values = sm.tsa.acf(data8['lnInflacio'], nlags= 10)
acf_values = acf_values[abs(acf_values) > 0.5] 

# Función PACF
pacf_values = sm.tsa.pacf(data8['lnInflacio'], nlags= 10)
pacf_values = pacf_values[abs(pacf_values) > 0.5] 

# Parámetros máximos MA y AC para la estimáción de un SARIMA - Logaritmo
MA_max = len(acf_values)
AC_max = len(pacf_values)

# Transformación en Cox-Box
# Función ACF
acf_values = sm.tsa.acf(data8['box_cox_Inflacion'], nlags= 10)
acf_values = acf_values[abs(acf_values) > 0.5] 

# Función PACF
pacf_values = sm.tsa.pacf(data8['box_cox_Inflacion'], nlags= 10)
pacf_values = pacf_values[abs(pacf_values) > 0.5] 

# Parámetros máximos MA y AC para la estimáción de un SARIMA - Logaritmo
MA_max_cb = len(acf_values)
AC_max_cb = len(pacf_values)

# =============================================================================
# Pruebas de Raíz Unitaria para la determinación del orden de integración
# =============================================================================


data8['Inflacion_1d'] = data8['Inflacion'].diff()

# Pruebas de estasionariedad: Primera diferencia

# Dickey-Fuller
dfp = adfuller(data8['Inflacion_1d'].dropna())
print('Dickey-Fuller p-value: %f' % dfp[1])

# Phillips-Perron
pp_test = PhillipsPerron(data8['Inflacion_1d'].dropna())
print(pp_test)

# Kwiatkowski-Phillips-Schmidt-Shin
kpss_test = kpss(data8['Inflacion_1d'].dropna())
print('KPSS p-value: %f' % kpss_test[1])

# Gráfico Serie Contemporanea 
sns.lineplot(x='Fecha', y='Inflacion_1d', data=data8)
plt.title('Inflación')
plt.show()

# Se concluye que la serie es estacionaria en primera diferencia, por tanto
# es una integrada de primer orden I(1)

# =============================================================================
# Modelos ARIMA
# =============================================================================

forecast_steps = 12

# Estimación con inflación en logaritmos 
m_arima_log = sm.tsa.ARIMA(data8['lnInflacio'], order=(2,1,3)) # 2,1,3
m_results_log = m_arima_log.fit()
m_results_log.summary()
residuos_log =  m_results_log.resid
estimados_log = m_results_log.fittedvalues

# Grafico de residuos
residuos_log.plot(title = 'Residuos del modelo ARIMA AR(1)I(1) MA(2) en logaritmos')
plt.show()

# Gráfico original vs ARIMA: Transformación logaritmica
plt.figure(figsize=(12, 6))
sns.lineplot(x=data8['Fecha'], y=data8['lnInflacio'], label='Inflación Observada', color='blue')
sns.lineplot(x=data8['Fecha'], y=estimados_log, label='Modelo Ajustado (ARIMA - [2,1,3])', linestyle='--', color='red')

# Configuración del gráfico
plt.title('Inflación Observada vs Modelo Ajustado', fontsize=16)
plt.xlabel('Fecha', fontsize=14)
plt.ylabel('Log de la Inflación', fontsize=14)
plt.legend(fontsize=12)
plt.grid(alpha=0.4)
plt.tight_layout()

# Mostrar el gráfico
plt.show()

# Evaluación de los residuo
from scipy.stats import shapiro, kstest, jarque_bera, probplot

# Normalidad de los residuos 
# Prueba de Shapiro-Wilk
stat, p = shapiro(residuos_log)
print('Prueba de Shapiro-Wilk:')
print('Estadístico =', stat, ', Valor p =', p)

# Prueba de Kolmogorov-Smirnov
stat, p = kstest(residuos_log, 'norm', args=(residuos_log.mean(), residuos_log.std()))
print('Prueba de Kolmogorov-Smirnov:')
print('Estadístico =', stat, ', Valor p =', p)

# Prueba de Jarque-Bera
stat, p = jarque_bera(residuos_log)
print('Prueba de Jarque-Bera:')
print('Estadístico =', stat, ', Valor p =', p)

# Prueba Ljung-Box de autocorrelación de los residuos
from statsmodels.stats.diagnostic import acorr_ljungbox

# Prueba de Ljung-Box
ljung_box_test = acorr_ljungbox(residuos_log, lags=[10], return_df=True)
print(ljung_box_test)
# Como lb_pvalue > 0.05, no hay evidencia suficiente para rechazar la hipótesis nula, lo que sugiere que 
# los residuos son independientes



# Estimación con inflación en Cox-box 
m_arima_bc = sm.tsa.ARIMA(data8['box_cox_Inflacion'], order=(2,1,3)) 
m_results_bc = m_arima_bc.fit()
m_results_bc.summary()
residuos_bc = m_results_bc.resid
estimados_bc = m_results_bc.fittedvalues

# Gráfico original vs ARIMA: Transformación Cox-box
plt.figure(figsize=(12, 6))
sns.lineplot(x=data8['Fecha'], y=data8['box_cox_Inflacion'], label='Inflación Observada', color='blue')
sns.lineplot(x=data8['Fecha'], y=estimados_bc, label='Modelo Ajustado (ARIMA - [2,1,3])', linestyle='--', color='red')

# Configuración del gráfico
plt.title('Inflación Observada vs Modelo Ajustado', fontsize=16)
plt.xlabel('Fecha', fontsize=14)
plt.ylabel('Log de la Inflación', fontsize=14)
plt.legend(fontsize=12)
plt.grid(alpha=0.4)
plt.tight_layout()

# Mostrar el gráfico
plt.show()

# Evaluación de los residuo
from scipy.stats import shapiro, kstest, jarque_bera, probplot

# Normalidad de los residuos 
# Prueba de Shapiro-Wilk
stat, p = shapiro(residuos_bc)
print('Prueba de Shapiro-Wilk:')
print('Estadístico =', stat, ', Valor p =', p)

# Prueba de Kolmogorov-Smirnov
stat, p = kstest(residuos_bc, 'norm', args=(residuos_bc.mean(), residuos_bc.std()))
print('Prueba de Kolmogorov-Smirnov:')
print('Estadístico =', stat, ', Valor p =', p)

# Prueba de Jarque-Bera
stat, p = jarque_bera(residuos_bc)
print('Prueba de Jarque-Bera:')
print('Estadístico =', stat, ', Valor p =', p)

# Prueba Ljung-Box de autocorrelación de los residuos
from statsmodels.stats.diagnostic import acorr_ljungbox

# Prueba de Ljung-Box
ljung_box_test_bc = acorr_ljungbox(residuos_bc, lags=[10], return_df=True)
print(ljung_box_test)
# Como lb_pvalue > 0.05, no hay evidencia suficiente para rechazar la hipótesis nula, lo que sugiere que 
# los residuos son independientes

# Metricas de error 

from sklearn.metrics import mean_squared_error


# Pronósticos

pronosticos_log = m_results_log.get_forecast(steps = forecast_steps)
pronosticos_bc = m_results_bc.get_forecast(steps = forecast_steps)

# Obtener predicciones y bandas de error
ajuste_medio_log = pronosticos_log.predicted_mean
conf_int_log = pronosticos_log.conf_int(alpha=0.05) 

ajuste_medio_bc = pronosticos_bc.predicted_mean
conf_int_bc = pronosticos_bc.conf_int(alpha=0.05) 

# Caso: Transformación Logarítmica
plt.figure(figsize=(10, 6))

# Datos históricos
plt.plot(data8['Fecha'], data8['lnInflacio'], label="Datos históricos", color="blue")

# Pronóstico
future_index = pd.date_range(start= '2024-12-01', end= '2025-12-01', freq = 'ME')
plt.plot(future_index, ajuste_medio_log, label="Pronóstico", color="green")

# Bandas de error
plt.fill_between(
    future_index,
    conf_int_log.iloc[:, 0],
    conf_int_log.iloc[:, 1],
    color="green",
    alpha=0.2,
    label="Intervalo de confianza 95%"
)

# Etiquetas y leyenda
plt.xlabel("Tiempo")
plt.ylabel("Valores")
plt.title("Pronóstico con bandas de error al 95%")
plt.legend()
plt.grid(True)

plt.show()

# Caso: Transformación Cox-Box
plt.figure(figsize=(10, 6))

# Datos históricos
plt.plot(data8['Fecha'], data8['box_cox_Inflacion'], label="Datos históricos", color="blue")

# Pronóstico
future_index = pd.date_range(start= '2024-12-01', end= '2025-12-01', freq = 'ME')
plt.plot(future_index, ajuste_medio_bc, label="Pronóstico", color="green")

# Bandas de error
plt.fill_between(
    future_index,
    conf_int_bc.iloc[:, 0],
    conf_int_bc.iloc[:, 1],
    color="green",
    alpha=0.2,
    label="Intervalo de confianza 95%"
)

# Etiquetas y leyenda
plt.xlabel("Tiempo")
plt.ylabel("Valores")
plt.title("Pronóstico con bandas de error al 95%")
plt.legend()
plt.grid(True)

plt.show()


# =============================================================================
# Modelos SARIMA
# =============================================================================

from statsmodels.tsa.statespace.sarimax import SARIMAX
from itertools import product

# Estimación con inflación en logaritmos 
m_sarima_log = SARIMAX(data8['lnInflacio'], order=(2,1,3), seasonal_order = (0,1,0,12)) # 2,1,3
sm_results_log = m_sarima_log.fit()
sm_results_log.summary()
sresiduos_log =  sm_results_log.resid
sestimados_log = sm_results_log.fittedvalues

# Grafico de residuos
residuos_log.plot(title = 'Residuos del modelo ARIMA AR(1)I(1) MA(2) en logaritmos')
plt.show()

# Gráfico original vs ARIMA: Transformación logaritmica
plt.figure(figsize=(12, 6))
sns.lineplot(x=data8['Fecha'], y=data8['lnInflacio'], label='Inflación Observada', color='blue')
sns.lineplot(x=data8['Fecha'], y=sestimados_log, label='Modelo Ajustado (ARIMA - [2,1,3])', linestyle='--', color='red')

# Configuración del gráfico
plt.title('Inflación Observada vs Modelo Ajustado', fontsize=16)
plt.xlabel('Fecha', fontsize=14)
plt.ylabel('Log de la Inflación', fontsize=14)
plt.legend(fontsize=12)
plt.grid(alpha=0.4)
plt.tight_layout()

# Mostrar el gráfico
plt.show()

# Evaluación de los residuo
from scipy.stats import shapiro, kstest, jarque_bera, probplot

# Normalidad de los residuos 
# Prueba de Shapiro-Wilk
stat, p = shapiro(sresiduos_log)
print('Prueba de Shapiro-Wilk:')
print('Estadístico =', stat, ', Valor p =', p)

# Prueba de Kolmogorov-Smirnov
stat, p = kstest(sresiduos_log, 'norm', args=(sresiduos_log.mean(), sresiduos_log.std()))
print('Prueba de Kolmogorov-Smirnov:')
print('Estadístico =', stat, ', Valor p =', p)

# Prueba de Jarque-Bera
stat, p = jarque_bera(sresiduos_log)
print('Prueba de Jarque-Bera:')
print('Estadístico =', stat, ', Valor p =', p)

# Prueba Ljung-Box de autocorrelación de los residuos
from statsmodels.stats.diagnostic import acorr_ljungbox

# Prueba de Ljung-Box
ljung_box_test = acorr_ljungbox(sresiduos_log, lags=[10], return_df=True)
print(ljung_box_test)
# Como lb_pvalue > 0.05, no hay evidencia suficiente para rechazar la hipótesis nula, lo que sugiere que 
# los residuos son independientes

# Pronósticos

pronosticos_log = sm_results_log.get_forecast(steps = forecast_steps)

# Obtener predicciones y bandas de error
ajuste_medio_log = pronosticos_log.predicted_mean
conf_int_log = pronosticos_log.conf_int(alpha=0.05)

# Caso: Transformación Logarítmica
plt.figure(figsize=(10, 6))

# Datos históricos
plt.plot(data8['Fecha'], data8['lnInflacio'], label="Datos históricos", color="blue")

# Pronóstico
future_index = pd.date_range(start= '2024-12-01', end= '2025-12-01', freq = 'ME')
plt.plot(future_index, ajuste_medio_log, label="Pronóstico", color="green")

# Bandas de error
plt.fill_between(
    future_index,
    conf_int_log.iloc[:, 0],
    conf_int_log.iloc[:, 1],
    color="green",
    alpha=0.2,
    label="Intervalo de confianza 95%"
)

# Etiquetas y leyenda
plt.xlabel("Tiempo")
plt.ylabel("Valores")
plt.title("Pronóstico con bandas de error al 95%")
plt.legend()
plt.grid(True)

plt.show()


# Estimación con inflación en Box-Cox

sm_arima_bc = SARIMAX(data8['box_cox_Inflacion'], order=(2,1,3), seasonal_order = (0,1,0,12)) # 2,1,3
sm_results_bc = sm_arima_bc.fit()
sm_results_bc.summary()
sresiduos_bc =  sm_results_bc.resid
sestimados_bc = sm_results_bc.fittedvalues

# Grafico de residuos
sresiduos_bc.plot(title = 'Residuos del modelo ARIMA AR(1)I(1) MA(2) en logaritmos')
plt.show()

# Gráfico original vs ARIMA: Transformación logaritmica
plt.figure(figsize=(12, 6))
sns.lineplot(x=data8['Fecha'], y=data8['box_cox_Inflacion'], label='Inflación Observada', color='blue')
sns.lineplot(x=data8['Fecha'], y=sestimados_bc, label='Modelo Ajustado (ARIMA - [2,1,3])', linestyle='--', color='red')

# Configuración del gráfico
plt.title('Inflación Observada vs Modelo Ajustado', fontsize=16)
plt.xlabel('Fecha', fontsize=14)
plt.ylabel('Log de la Inflación', fontsize=14)
plt.legend(fontsize=12)
plt.grid(alpha=0.4)
plt.tight_layout()

# Mostrar el gráfico
plt.show()

# Evaluación de los residuo
from scipy.stats import shapiro, kstest, jarque_bera, probplot

# Normalidad de los residuos 
# Prueba de Shapiro-Wilk
stat, p = shapiro(sresiduos_bc)
print('Prueba de Shapiro-Wilk:')
print('Estadístico =', stat, ', Valor p =', p)

# Prueba de Kolmogorov-Smirnov
stat, p = kstest(sresiduos_bc, 'norm', args=(sresiduos_bc.mean(), sresiduos_bc.std()))
print('Prueba de Kolmogorov-Smirnov:')
print('Estadístico =', stat, ', Valor p =', p)

# Prueba de Jarque-Bera
stat, p = jarque_bera(sresiduos_bc)
print('Prueba de Jarque-Bera:')
print('Estadístico =', stat, ', Valor p =', p)

# Prueba Ljung-Box de autocorrelación de los residuos
from statsmodels.stats.diagnostic import acorr_ljungbox

# Prueba de Ljung-Box
ljung_box_test = acorr_ljungbox(sresiduos_bc, lags=[10], return_df=True)
print(ljung_box_test)
# Como lb_pvalue > 0.05, no hay evidencia suficiente para rechazar la hipótesis nula, lo que sugiere que 
# los residuos son independientes

# Pronósticos

spronosticos_bc = sm_results_bc.get_forecast(steps = forecast_steps)

# Obtener predicciones y bandas de error
sajuste_medio_bc = spronosticos_bc.predicted_mean
sconf_int_bc = spronosticos_bc.conf_int(alpha=0.05)

# Caso: Transformación Logarítmica
plt.figure(figsize=(10, 6))

# Datos históricos
plt.plot(data8['Fecha'], data8['box_cox_Inflacion'], label="Datos históricos", color="blue")

# Pronóstico
future_index = pd.date_range(start= '2024-12-01', end= '2025-12-01', freq = 'ME')
plt.plot(future_index, sajuste_medio_bc, label="Pronóstico", color="green")

# Bandas de error
plt.fill_between(
    future_index,
    sconf_int_bc.iloc[:, 0],
    sconf_int_bc.iloc[:, 1],
    color="green",
    alpha=0.2,
    label="Intervalo de confianza 95%"
)

# Etiquetas y leyenda
plt.xlabel("Tiempo")
plt.ylabel("Valores")
plt.title("Pronóstico con bandas de error al 95%")
plt.legend()
plt.grid(True)

plt.show()


# Resultados 
# Funcion inversa de Box-Co

resultados_log = np.exp(ajuste_medio_log)
resultados_bc = np.power((lambda_ * ajuste_medio_bc) + 1, 1 / lambda_)

