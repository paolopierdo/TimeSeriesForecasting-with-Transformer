#Importing libraries
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import requests
import statsmodels.api as sm
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
from statsmodels.datasets import get_rdataset
from datetime import timedelta
import os
import plotly.io as pio
pio.renderers.default = "notebook"
import matplotlib.pyplot as plt
import optuna
import xgboost as xgb
import seaborn as sns
import pytorch_lightning as pl
from neuralforecast import NeuralForecast
from neuralforecast.models import TFT, LSTM
from neuralforecast.losses.pytorch import DistributionLoss
from neuralforecast.utils import AirPassengersPanel, AirPassengersStatic, AirPassengersDF
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import MinMaxScaler
from sklearn.impute import KNNImputer
from sklearn.metrics import mean_absolute_error, mean_squared_error, mean_absolute_percentage_error
from sklearn.model_selection import cross_val_score, KFold, TimeSeriesSplit
import fastf1
from scipy.stats import pearsonr
import random
import torch
import torch.nn as nn
import matplotlib as mpl
import matplotlib.gridspec as gridspec
from statsmodels.nonparametric.smoothers_lowess import lowess
from statsmodels.tsa.stattools import adfuller, kpss
from statsmodels.tsa.seasonal import seasonal_decompose, STL
from statsmodels.tsa.arima.model import ARIMA
from statsforecast import StatsForecast
from statsforecast.models import AutoARIMA, AutoETS
from statsforecast.arima import arima_string
import scipy.stats as stats
from scipy import stats
from functools import partial
import utilsforecast.losses as ufl
from utilsforecast.evaluation import evaluate
import warnings
warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=UserWarning)
from neuralprophet import NeuralProphet
from time import time
from optuna.importance import get_param_importances
import optuna.visualization as vis
import pickle
from sklearn.model_selection import ParameterGrid
from tqdm import tqdm
from autogluon.timeseries import TimeSeriesDataFrame, TimeSeriesPredictor
from pytorch_lightning.callbacks import ModelCheckpoint, LearningRateMonitor
from pytorch_lightning.loggers import TensorBoardLogger

###################### Importing necessary data #############################
# Mappa dei dataset: nome_variabile -> (percorso, parametri opzionali)
#data_folder = "/Volumes/club_ai/tstrans/tsdata" # per databricks
data_folder = 'C:/Users/ppierdomenico/OneDrive - ICONSULTING S.p.A/Desktop/ICO-PAPI/timeseries_ai/data'
all_datasets = {
    "WN": (f"{data_folder}/wn.csv", {}),
    "EngTemp": (f"{data_folder}/engTemp.csv", {}),
    "oil_prices": (f"{data_folder}/oil2.csv", {"index_col":"date","parse_dates":True}),
    "nile": (f"{data_folder}/Nile.csv", {}),
    "coviduk": (f"{data_folder}/covid_uk.csv", {}),
    "missing": (f"{data_folder}/missing.csv", {}),
    "gold": (f"{data_folder}/gold.csv", {}),
    "solar_power": (f"{data_folder}/SolarData.csv", {}),
    "bit": (f"{data_folder}/Bitbay_BTCUSD_1h.csv", {}),
    "aqi": (f"{data_folder}/AirQualityUCI.csv", {"sep":";"}),
    "f1": (f"{data_folder}/F1_data.csv", {}),
    "elec": (f"{data_folder}/ElecDemand.csv", {}),
    "favorita_train": (f"{data_folder}/FAVORITA_train.csv", {}),
    "favorita_test": (f"{data_folder}/FAVORITA_test.csv", {}),
    "rossman_store": (f"{data_folder}/ROSSMAN_store.csv", {}),
    "rossman_train": (f"{data_folder}/ROSSMAN_train.csv", {"index_col": "Date", "parse_dates": True}),
    "rossman_test": (f"{data_folder}/ROSSMAN_test.csv", {}),
    "walmart_features": (f"{data_folder}/WALMART_features.csv", {}),
    "walmart_stores": (f"{data_folder}/WALMART_stores.csv", {}),
    "walmart_train": (f"{data_folder}/WALMART_train.csv", {}),
    "walmart_test": (f"{data_folder}/WALMART_test.csv", {}),
    'daily_oil':(f"{data_folder}/daily_oil.csv",{}),
    'combined_df_LSTM':(f"{data_folder}/combined_df_LSTM.csv",{})
}
selected_datasets = ["oil_prices", "solar_power", "f1", "elec", 
                     "favorita_train", "favorita_test", 'daily_oil']
selected_data = {key: all_datasets[key] for key in selected_datasets}
# Reading data from files in data directory
loaded_data = {}
for name, (path, params) in selected_data.items():
    try: loaded_data[name] = pd.read_csv(path, **params)
    except Exception as e: print(f"Error while loading {name}: {e}") 
# Defining functions

# Plotting function
def ts_plotly(series_dict, labels=None, title="Time series"):
    fig = go.Figure()
    for label, series in series_dict.items():
        fig.add_trace(go.Scatter(x=series.index, y=series.values, mode='lines', name=label))
    fig.update_layout(title=title, xaxis_title="Data", yaxis_title="Value", template="plotly_dark")
    fig.show()
    
########################## Funzioni per test stazionarietà ################################
# Test Augmented Dickey-Fuller
def test_adf(series):
    result = adfuller(series.dropna())
    print('\nL"ipotesi nulla del Test Dickey-Fuller è H0: SERIE NON STAZIONARIA')
    print('Test Augmented Dickey-Fuller:')
    print(f'Statistica del test: {result[0]:.4f}')
    print(f'p-value: {result[1]:.4f}')
    print(f'Valori critici:')
    for key, value in result[4].items():
        print(f'\t{key}: {value:.4f}')
    if result[1] < 0.05:
        print("Conclusione: La serie è stazionaria (rifiuto dell'ipotesi nulla per alpha=0.05)")
    else:
        print("Conclusione: La serie non è stazionaria (non posso rifiutare l'ipotesi nulla per alpha=0.05)")
    
# Test KPSS
def test_kpss(series):
    result = kpss(series.dropna())
    print('\nL"ipotesi nulla del Test KPSS è H0: SERIE STAZIONARIA')
    print('Test KPSS:')
    print(f'Statistica del test: {result[0]:.4f}')
    print(f'p-value: {result[1]:.4f}')
    print(f'Valori critici:')
    for key, value in result[3].items():
        print(f'\t{key}: {value:.4f}')
    if result[1] < 0.05:
        print("Conclusione: La serie non è stazionaria (rifiuto dell'ipotesi nulla per alpha=0.05)")
    else:
        print("Conclusione: La serie è stazionaria (non posso rifiutare l'ipotesi nulla per alpha=0.05)")
        
############################ Funzione per decomposizione additiva e moltiplicativa ################
def plot_decomposition(decomposition_add, decomposition_mult):
    components = ['observed', 'trend', 'seasonal', 'resid']
    titles = ['Serie originale', 'Trend', 'Stagionalità', 'Residui']

    plt.figure(figsize=(16, 12))

    for i, (comp, title) in enumerate(zip(components, titles)):
        # Colonna sinistra: Additiva
        plt.subplot(4, 2, 2*i + 1)
        plt.plot(getattr(decomposition_add, comp), label='Additiva')
        plt.title(f'Additiva - {title}')
        plt.legend(loc='best')

        # Colonna destra: Moltiplicativa
        plt.subplot(4, 2, 2*i + 2)
        plt.plot(getattr(decomposition_mult, comp), label='Moltiplicativa')
        plt.title(f'Moltiplicativa - {title}')
        plt.legend(loc='best')

    plt.tight_layout()
    plt.show()
    
############################ Funzioni per metriche di valutazione ########################à
# MAPE
def mape(y_true, y_pred):
    y_true, y_pred = np.array(y_true), np.array(y_pred)
    # Evita divisione per zero
    mask = y_true != 0
    return np.mean(np.abs((y_true[mask] - y_pred[mask]) / y_true[mask])) * 100

# SMAPE
def smape(y_true, y_pred):
    y_true, y_pred = np.array(y_true), np.array(y_pred)
    denominator = (np.abs(y_true) + np.abs(y_pred))
    # Evita divisione per zero
    mask = denominator != 0
    diff = np.abs(y_true[mask] - y_pred[mask]) / denominator[mask]
    return np.mean(diff) * 100

# RMSE
def calcola_rmse(y_true, y_pred):
    y_true, y_pred = np.array(y_true), np.array(y_pred)
    return np.sqrt(np.mean((y_true - y_pred) ** 2))

# MAE
def calcola_mae(y_true, y_pred):
    y_true, y_pred = np.array(y_true), np.array(y_pred)
    return np.mean(np.abs(y_true - y_pred))

# MAE STD
def mae_std(training_series, testing_series, forecast_series):
    d = np.std(training_series)
    errors = np.abs(testing_series - forecast_series)
    if d == 0:
        return np.nan
    return errors.mean() / d
    
# MASE (Naive)
def mase(training_series, testing_series, forecast_series):
    naive_diff = np.abs(np.diff(training_series))
    d = naive_diff.mean()
    errors = np.abs(testing_series.values - forecast_series.values) #prima era senza .values
    if d == 0:
        return np.nan
    return errors.mean() / d

# WQL
def weighted_quantile_loss(y_true, y_pred_quantiles, quantiles):
    y_true = np.asarray(y_true)
    
    # Converti y_pred_quantiles in un formato adatto
    if isinstance(y_pred_quantiles, pd.DataFrame):
        # Estrai le colonne dei quantili dal DataFrame
        quantile_preds = {float(q): y_pred_quantiles[str(q)].values for q in quantiles}
    elif isinstance(y_pred_quantiles, dict):
        quantile_preds = y_pred_quantiles
    else:
        raise ValueError("y_pred_quantiles deve essere un DataFrame o un dizionario")
    
    wql_per_quantile = {}    
    for q in quantiles:
        y_pred_q = np.asarray(quantile_preds[q])
        
        # Verifica che le dimensioni corrispondano
        if len(y_true) != len(y_pred_q):
            raise ValueError(f"Dimensioni incompatibili: y_true ({len(y_true)}) e y_pred_q ({len(y_pred_q)})")
        
        # Calcola gli errori
        errors = y_true - y_pred_q
        
        # Applica la funzione di perdita quantile
        # Se error > 0 (sottostima), moltiplica per q
        # Se error <= 0 (sovrastima), moltiplica per (1-q)
        quantile_losses = np.where(
            errors >= 0,
            q * np.abs(errors),
            (1 - q) * np.abs(errors)
        )
        
        # Media delle perdite per questo quantile
        wql_q = np.mean(quantile_losses)
        wql_per_quantile[q] = wql_q  
    return wql_per_quantile
# WQL 2

    """
    Calcola la Weighted Quantile Loss correttamente bilanciata.
    
    Parameters:
    -----------
    y_true : array-like
        Valori osservati
    y_pred_quantiles : dict o DataFrame
        Previsioni dei quantili
    quantiles : list
        Lista dei quantili da valutare
    
    Returns:
    --------
    dict
        WQL per ogni quantile e WQL media
    """
    y_true = np.asarray(y_true).flatten()
    
    # Converti y_pred_quantiles in un formato adatto
    if isinstance(y_pred_quantiles, pd.DataFrame):
        # Estrai le colonne dei quantili dal DataFrame
        quantile_preds = {float(q): y_pred_quantiles[str(q)].values.flatten() for q in quantiles}
    elif isinstance(y_pred_quantiles, dict):
        quantile_preds = {q: np.asarray(v).flatten() for q, v in y_pred_quantiles.items()}
    else:
        raise ValueError("y_pred_quantiles deve essere un DataFrame o un dizionario")
    
    # Verifica che i quantili siano validi
    for q in quantiles:
        if q <= 0 or q >= 1:
            raise ValueError(f"I quantili devono essere compresi tra 0 e 1, trovato {q}")
    
    wql_per_quantile = {}
    total_wql = 0
    
    for q in quantiles:
        y_pred_q = quantile_preds[q]
        
        # Verifica che le dimensioni corrispondano
        if len(y_true) != len(y_pred_q):
            raise ValueError(f"Dimensioni incompatibili: y_true ({len(y_true)}) e y_pred_q ({len(y_pred_q)})")
        
        # Calcola gli errori
        errors = y_true - y_pred_q
        
        # Applica la funzione di perdita quantile
        quantile_losses = np.where(
            errors >= 0,
            q * errors,         # Sottostima (valore reale > previsto)
            (1 - q) * (-errors)  # Sovrastima (valore reale < previsto)
        )
        
        # Media delle perdite per questo quantile
        wql_q = np.mean(quantile_losses)
        wql_per_quantile[q] = wql_q
        total_wql += wql_q
    
    # Media delle WQL per tutti i quantili
    wql_per_quantile['mean'] = total_wql / len(quantiles)
    
    return wql_per_quantile
# calcola WQL
def calcola_wql(y_true, y_pred_quantiles, quantiles=None):
    # Calcola la WQL
    wql_per_quantile = weighted_quantile_loss(y_true, y_pred_quantiles, quantiles)
    # Prepara il risultato
    result = {'wql_per_quantile': wql_per_quantile}
    return result

# Funzione generale (che le chiama tutte)
def calcola_metriche(y_true, y_pred, y_train=None, modelname=None, y_pred_quantiles=None, quantiles=None):
    metriche = {
        'mae': calcola_mae(y_true, y_pred),
        'rmse': calcola_rmse(y_true, y_pred),
        'mape': mape(y_true, y_pred),
        'smape': smape(y_true, y_pred),
        #'mae_std': mae_std(y_train, y_true, y_pred),
        'mase': mase(y_train, y_true, y_pred)
    }
        # Aggiungi WQL se sono fornite le previsioni quantili
    if y_pred_quantiles is not None:
        wql_results = calcola_wql(y_true, y_pred_quantiles, quantiles)
        for q, loss in wql_results['wql_per_quantile'].items():
            metriche[f'wql_{q}'] = loss
            
    # Crea un dataframe con le metriche
    df_metriche = pd.DataFrame(metriche, index=['valore']).T
    df_metriche.columns = [modelname]
    return df_metriche

######################### Funzione per plottare i residui #################################à
def plot_residuals(residuals, model_name="Model"):
    residual_series = pd.Series(residuals).astype(float).dropna()
    residual_df = pd.DataFrame({'residual': residual_series})
    fig, axs = plt.subplots(nrows=2, ncols=2, figsize=(12, 8))
    print(f"---------Residui per {model_name}---------")
    # 1. Line plot
    residual_df.plot(ax=axs[0,0])
    axs[0,0].set_title("Residuals")
    # 2. Density
    sns.histplot(residual_df["residual"], ax=axs[0,1], kde=True)
    axs[0,1].set_title("Density plot")
    # 3. Q-Q plot
    stats.probplot(residual_df["residual"], dist="norm", plot=axs[1,0])
    axs[1,0].set_title("Q-Q plot")
    # 4. ACF
    plot_acf(residual_df["residual"], lags=35, ax=axs[1,1], color="fuchsia")
    axs[1,1].set_title("Autocorrelation")
    plt.tight_layout()
    plt.show()