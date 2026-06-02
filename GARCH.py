import matplotlib.pyplot as plt
from arch import arch_model
from sklearn.metrics import mean_squared_error

from preprocessing import df_sip, df_wig


def _get_splits():
    train_sip = df_sip.loc[df_sip.index < "2025-01-01"]
    test_sip = df_sip.loc[df_sip.index >= "2025-01-01"]
    train_wig = df_wig.loc[df_wig.index < "2025-01-01"]
    test_wig = df_wig.loc[df_wig.index >= "2025-01-01"]
    return train_sip, test_sip, train_wig, test_wig


def run_garch():
    # Get train/test splits
    train_sip, test_sip, train_wig, test_wig = _get_splits()
    # Przekształcamy zwroty na procenty (arch zwykle pracuje na % zwrotach)
    train_sip_return = train_sip["log_return"] * 100
    train_wig_return = train_wig["log_return"] * 100
    test_sip_return = test_sip["log_return"] * 100
    test_wig_return = test_wig["log_return"] * 100

    model_sip = arch_model(train_sip_return, vol="Garch", p=1, q=1)
    model_fit_sip = model_sip.fit()
    print(model_fit_sip.summary())

    model_wig = arch_model(train_wig_return, vol="Garch", p=1, q=1)
    model_fit_wig = model_wig.fit()
    print(model_fit_wig.summary())

    forecast_sip = model_fit_sip.forecast(horizon=len(test_sip_return))
    forecast_wig = model_fit_wig.forecast(horizon=len(test_wig_return))
    print(forecast_sip)
    print(forecast_wig)

    # Metryki: porównujemy prognozowaną wariancję (forecast.variance) z kwadratami rzeczywistych zwrotów
    # Uwaga: kwadraty zwrotów są prostym proxy wariancji; są bardzo szumne, więc metryki RMSE/MSE
    # mogą być duże. Rozważ również agregację/filtrację przed porównaniem.
    var_sip = forecast_sip.variance.values[-1, :]
    var_wig = forecast_wig.variance.values[-1, :]

    mse_sip = mean_squared_error((test_sip_return ** 2), var_sip)
    rmse_sip = (mse_sip ** 0.5)

    print("S&P MSE GARCH:", mse_sip)
    print("S&P RMSE GARCH:", rmse_sip)

    mse_wig = mean_squared_error((test_wig_return ** 2), var_wig)
    rmse_wig = (mse_wig ** 0.5)

    print("WiG MSE GARCH:", mse_wig)
    print("WiG RMSE GARCH:", rmse_wig)

    plt.figure(figsize=(12,6))
    plt.plot(test_sip_return.index, test_sip_return ** 2, label="Real")
    plt.plot(test_sip_return.index, var_sip, label="Forecast")
    plt.legend()
    plt.title("GARCH Forecast vs Real Values for S&P 500")
    plt.show()

    plt.figure(figsize=(12,6))
    plt.plot(test_wig_return.index, test_wig_return ** 2, label="Real")
    plt.plot(test_wig_return.index, var_wig, label="Forecast")
    plt.legend()
    plt.title("GARCH Forecast vs Real Values for WIG 20")
    plt.show()


if __name__ == "__main__":
    run_garch()
