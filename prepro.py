# QUESTO FILE SERVE PER SALVARE TUTTI I PREPROCESSING CHE HO FATTO
# NEI PROSSIMI NOTEBOOK, DEI MODELLI, LANCIO QUESTO E SIAMO PRONTI
from my_import import *
print(loaded_data.keys()) #tutti i dataset
series_dict = {}
refresh_data = False


# %% [markdown]
# ### **WHITE NOISE**

# %%
#generate data
n_years = 10
start_date = "2014-01-01"
series_name = "White Noise"
date_range = pd.date_range(start=start_date, periods=365*n_years, freq='D')
np.random.seed(42)
white_noise = np.random.normal(loc=0, scale=1, size=len(date_range))
series = pd.Series(white_noise, index=date_range)
series_dict[series_name] = series
# --------------------series-------------------------

# %% [markdown]
# Questa serie è pronta così, non c'è bisogno di toccarla


# %% [markdown]
# ### **OIL PRICE**
# %%
oil_prices = loaded_data['oil_prices']
oil_prices.reset_index(inplace=True)
oil_prices['date'] = oil_prices['date'].dt.strftime('%Y-%m')
oil_prices.set_index('date', inplace=True)
# --------------------oil_prices-------------------------

# %% [markdown]
# ### **missing data**: serie univariata delle sales di Favorita
# %%
missing = loaded_data['favorita_train']
missing.set_index("date", inplace=True)
missing = missing[['sales','family']]
# %%
#sommo tutte le vendite di ogni store per ogni giorno
missing = missing.groupby(['date']).sum().reset_index()
missing.drop(columns=["family"], inplace=True)
missing = missing.rename(columns={"sales": "total_daily_sales"})
missing.set_index("date", inplace=True)

# %%
#introduciamo un 20% di NaN randomico con set_seed(24)
np.random.seed(24)
nan_indices = np.random.choice(missing.index, size=int(len(missing) * 0.2), replace=False)
missdata = missing.copy()
missdata.loc[nan_indices, 'total_daily_sales'] = np.nan
# --------------------missdata-------------------------

# %% [markdown]
# ### **outliers data**: serie univariata delle sales di Favorita
# %%
outl = missing.copy()
# %%
def create_outliers():
    # Creo qualche outlier in maniera randomica
    np.random.seed(22)
    # Genera 7 valori random tra 2,000,000 e 4,000,000
    outliers_positive = np.random.randint(2000000, 4000000, 7)
    # Genera 4 valori random tra -2,000,000 e -4,000,000
    outliers_negative = np.random.randint(-3000000, -1000000, 4)
    # Combina gli outlier positivi e negativi
    outliers = np.concatenate((outliers_positive, outliers_negative))
    return outliers
# %%
# Crea gli outlier
outliers = create_outliers()
# Sostituisci 11 valori casuali nel DataFrame con gli outlier
np.random.seed(22)
random_indices = np.random.choice(outl.size, 11, replace=False)
flat_df = outl.values.flatten()
flat_df[random_indices] = outliers
out = pd.DataFrame(flat_df.reshape(outl.shape), columns=outl.columns, index = missdata.index)
# --------------------out-------------------------

# %% [markdown]
# ### **SOLAR ENERGY**
# %%
SolarData = loaded_data['solar_power']
SolarData.set_index('DATE_TIME',inplace=True)

# %% [markdown]
# aggiungo l'ora del giorno
# %%
df = SolarData.copy()
df_reset = df.reset_index()
df_reset['DATE_TIME'] = pd.to_datetime(df_reset['DATE_TIME'])
# NUOVE VARIABILI TEMPORALI
df_reset['giorno_settimana'] = df_reset['DATE_TIME'].dt.dayofweek + 1
df_reset['ora_del_giorno'] = df_reset['DATE_TIME'].dt.hour + 1
#df_reset['settimana_del_mese'] = ((df_reset['DATE_TIME'].dt.day - 1) // 7) + 1
#df_reset['settimana_del_anno'] = df_reset['DATE_TIME'].dt.isocalendar().week
#df_reset['giorno_del_mese'] = df_reset['DATE_TIME'].dt.day
#df_reset['mese_del_anno'] = df_reset['DATE_TIME'].dt.month
df_solar = df_reset.set_index('DATE_TIME')
# --------------------df_solar-------------------------


# %% [markdown]
# ### **F1 RACE DATA**
# %%
f1 = loaded_data['f1']
f1.set_index("Date",inplace=True)
# %%
f1.sort_index(inplace=True)
# --------------------f1-------------------------


# %% [markdown]
# ### **ELECTRICTY DEMAND**
# %%
elec = loaded_data['elec']
# %%
def add_midnight_time(date_str):
    if len(date_str) < 19:  # Se la stringa è più corta di "2021-01-01 00:00:00"
        return date_str + " 00:00:00"
    return date_str
elec['Date'] = elec['Date'].apply(add_midnight_time)
elec['Date'] = pd.to_datetime(elec['Date'])
elec.set_index('Date',inplace=True)
# %%
#adding features
df_reset = elec.reset_index()
df_reset.rename(columns={'Date':'DATE_TIME'}, inplace=True)
df_reset['DATE_TIME'] = pd.to_datetime(df_reset['DATE_TIME'])
df_reset['giorno_settimana'] = df_reset['DATE_TIME'].dt.dayofweek + 1
df_reset['ora_del_giorno'] = df_reset['DATE_TIME'].dt.hour + 1
df_reset['settimana_del_mese'] = ((df_reset['DATE_TIME'].dt.day - 1) // 7) + 1
df_reset['settimana_del_anno'] = df_reset['DATE_TIME'].dt.isocalendar().week
df_reset['giorno_del_mese'] = df_reset['DATE_TIME'].dt.day
df_reset['mese_del_anno'] = df_reset['DATE_TIME'].dt.month
df_elec = df_reset.set_index('DATE_TIME')
# --------------------df_elec-------------------------

# %% [markdown]
# ### **FAVORITA**
# %%
favorita_train = loaded_data['favorita_train']
#favorita_train.set_index("date", inplace=True)
favorita_train.drop(columns=['id'],inplace=True)
favorita_train.rename(columns={"store_nbr": "Store","family":"Product_type"}, inplace=True)
# %%
#Creo una dummy per le promozione
favorita_train['promo'] = favorita_train['onpromotion'].apply(lambda x: 0 if x == 0 else 1)
favorita_train = favorita_train.reset_index().groupby(['date', 'Store'], as_index=False).agg({
    'sales': 'sum',
    'onpromotion': 'sum',    
    'promo': 'first'
})
#voglio aggiungere il prezzo del petrolio come regressore
oil = loaded_data['daily_oil']
oil.fillna(0,inplace=True)
# %%
favorita_train = pd.merge(favorita_train, oil, on='date',how='left')
# %%
favorita_train.rename(columns={"dcoilwtico":"oil price"},inplace=True)
favorita_train.sort_values(by="Store",inplace=True)
df_reset = favorita_train.reset_index()
df_reset.rename(columns={'date':'DATE_TIME'}, inplace=True)
df_reset['DATE_TIME'] = pd.to_datetime(df_reset['DATE_TIME'])
df_reset['giorno_settimana'] = df_reset['DATE_TIME'].dt.dayofweek + 1
df_reset['settimana_del_mese'] = ((df_reset['DATE_TIME'].dt.day - 1) // 7) + 1
df_reset['settimana_del_anno'] = df_reset['DATE_TIME'].dt.isocalendar().week
df_reset['giorno_del_mese'] = df_reset['DATE_TIME'].dt.day
df_reset['mese_del_anno'] = df_reset['DATE_TIME'].dt.month
favorita_train = df_reset.set_index('DATE_TIME')
favorita_train['rolling_mean_30'] = favorita_train.groupby('Store')['sales'].transform(lambda x: x.rolling(30, min_periods=1).mean())

favorita_train.reset_index(inplace=True)
favorita_train = favorita_train.sort_values(['Store','DATE_TIME'])
for lag in [1, 2, 7, 14, 28]:
    favorita_train[f'sales_lag_{lag}'] = favorita_train.groupby('Store')['sales'].shift(lag)
favorita_train.set_index('DATE_TIME',inplace=True)