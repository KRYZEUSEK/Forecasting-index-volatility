import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error
from statsmodels.tsa.arima.model import ARIMA

from preprocessing import df_sip, df_wig


def run_arima():
    train_sip = df_sip.loc[df_sip.index < "2025-01-01"]
    test_sip = df_sip.loc[df_sip.index >= "2025-01-01"]
    train_wig = df_wig.loc[df_wig.index < "2025-01-01"]
    test_wig = df_wig.loc[df_wig.index >= "2025-01-01"]

    model_sip = ARIMA(train_sip["log_return"], order=(1, 0, 0))
    model_fit_sip = model_sip.fit()
    print(model_fit_sip.summary())

    model_wig = ARIMA(train_wig["log_return"], order=(1, 0, 0))
    model_fit_wig = model_wig.fit()
    print(model_fit_wig.summary())

    forecast_sip = model_fit_sip.forecast(steps=len(test_sip["log_return"]))
    forecast_wig = model_fit_wig.forecast(steps=len(test_wig["log_return"]))
    print(forecast_sip)
    print(forecast_wig)

    mse_sip = mean_squared_error(test_sip["log_return"], forecast_sip)
    rmse_sip = (mse_sip ** 0.5)

    print("S&P MSE ARIMA:", mse_sip)
    print("S&P RMSE ARIMA:", rmse_sip)

    mse_wig = mean_squared_error(test_wig["log_return"], forecast_wig)
    rmse_wig = (mse_wig ** 0.5)

    print("WiG MSE ARIMA:", mse_wig)
    print("WiG RMSE ARIMA:", rmse_wig)

    plt.figure(figsize=(12,6))
    plt.plot(test_sip["log_return"].index, test_sip["log_return"], label="Real")
    plt.plot(test_sip["log_return"].index, forecast_sip, label="Forecast")
    plt.legend()
    plt.title("ARIMA Forecast vs Real Values for S&P 500")
    plt.show()

    plt.figure(figsize=(12,6))
    plt.plot(test_wig["log_return"].index, test_wig["log_return"], label="Real")
    plt.plot(test_wig["log_return"].index, forecast_wig, label="Forecast")
    plt.legend()
    plt.title("ARIMA Forecast vs Real Values for WIG 20")
    plt.show()


if __name__ == "__main__":
    run_arima()
