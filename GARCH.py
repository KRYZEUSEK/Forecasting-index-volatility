import matplotlib.pyplot as plt
from arch import arch_model
from sklearn.metrics import mean_squared_error, root_mean_squared_error

from ARIMA import test_sip, test_wig, train_sip, train_wig

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

mse_sip = mean_squared_error(test_sip_return**2, forecast_sip.variance.values[-1, :])
rmse_sip = root_mean_squared_error(test_sip_return**2, forecast_sip.variance.values[-1, :])

print("S&P MSE:", mse_sip)
print("S&P RMSE:", rmse_sip)

mse_wig = mean_squared_error(test_wig_return**2, forecast_wig.variance.values[-1, :])
rmse_wig = root_mean_squared_error(test_wig_return**2, forecast_wig.variance.values[-1, :])

print("WiG MSE:", mse_wig)
print("WiG RMSE:", rmse_wig)

plt.figure(figsize=(12,6))
plt.plot(test_sip_return.index, test_sip_return**2, label="Real")
plt.plot(test_sip_return.index, forecast_sip.variance.values[-1], label="Forecast")
plt.legend()
plt.title("GARCH Forecast vs Real Values for S&P 500")
plt.show()

plt.figure(figsize=(12,6))
plt.plot(test_wig_return.index, test_wig_return**2, label="Real")
plt.plot(test_wig_return.index, forecast_wig.variance.values[-1], label="Forecast")
plt.legend()
plt.title("GARCH Forecast vs Real Values for WIG 20")
plt.show()