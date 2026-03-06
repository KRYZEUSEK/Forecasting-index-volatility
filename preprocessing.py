import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
from statsmodels.stats.diagnostic import het_arch
from statsmodels.tsa.stattools import adfuller

df_sip = pd.read_csv(
    "C:/Users/niemi/OneDrive - Uniwersytet Ekonomiczny w Poznaniu/Pulpit/Datasets/^spx_d.csv"
)
df_wig = pd.read_csv(
    "C:/Users/niemi/OneDrive - Uniwersytet Ekonomiczny w Poznaniu/Pulpit/Datasets/wig20_d.csv"
)

print(df_wig.isnull().sum())
print(df_sip.isnull().sum())

print(df_wig.isna().sum())
print(df_sip.isna().sum())

pd.to_datetime(df_sip["Data"])
pd.to_datetime(df_wig["Data"])

df_sip = df_sip.set_index("Data")
df_wig = df_wig.set_index("Data")

df_sip = df_sip.sort_index()
df_wig = df_wig.sort_index()

df_sip.index = pd.to_datetime(df_sip.index)
df_sip = df_sip.asfreq('B')

df_wig.index = pd.to_datetime(df_wig.index)
df_wig = df_wig.asfreq('B')

print(df_sip.dtypes)
print(df_wig.dtypes)

df_sip["log_return"] = np.log(df_sip["Zamkniecie"] / df_sip["Zamkniecie"].shift(1))
df_wig["log_return"] = np.log(df_wig["Zamkniecie"] / df_wig["Zamkniecie"].shift(1))

df_sip = df_sip.dropna()
df_wig = df_wig.dropna()

adf_sip = adfuller(df_sip["log_return"])
adf_wig = adfuller(df_wig["log_return"])
print(adf_sip)
print(adf_wig)
#szeregi są stacjonarne 

plot_acf(df_sip["log_return"])
plot_pacf(df_sip["log_return"])

plot_acf(df_wig["log_return"])
plot_pacf(df_wig["log_return"])

plt.show()

arch_sip = het_arch(df_sip["log_return"])
arch_wig = het_arch(df_wig["log_return"])
print(arch_sip)
print(arch_wig)
# szeregi są bardzo heteroskedastyczne, więc model ARIMA będziemy zastępować GARCH.
print(df_sip.head())
print(df_wig.head())