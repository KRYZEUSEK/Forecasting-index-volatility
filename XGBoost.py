import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error, root_mean_squared_error
from xgboost import XGBRegressor, plot_importance

from NeuralNetwork import (X_test_sip, X_test_wig, X_train_sip, X_train_wig,
                           y_test_sip, y_test_wig, y_train_sip, y_train_wig)

model_sip = XGBRegressor(
    n_estimators=300,
    max_depth=4,
    learning_rate=0.05,
    subsample=0.8,
    colsample_bytree=0.8,
    random_state=42,
)

model_sip.fit(X_train_sip, y_train_sip,
              eval_set=[(X_test_sip, y_test_sip)],
              verbose=False)

pred_sip = model_sip.predict(X_test_sip)

model_wig = XGBRegressor(
    n_estimators=300,
    max_depth=4,
    learning_rate=0.05,
    subsample=0.8,
    colsample_bytree=0.8,
    random_state=42,
)

model_wig.fit(X_train_wig, y_train_wig,
              eval_set=[(X_test_wig, y_test_wig)], 
              verbose=False)

pred_wig = model_wig.predict(X_test_wig)

mse_sip = mean_squared_error(y_test_sip, pred_sip)
rmse_sip = root_mean_squared_error(y_test_sip, pred_sip)

print("S&P MSE:", mse_sip)
print("S&P RMSE:", rmse_sip)

mse_wig = mean_squared_error(y_test_wig, pred_wig)
rmse_wig = root_mean_squared_error(y_test_wig, pred_wig)

print("WiG MSE:", mse_wig)
print("WiG RMSE:", rmse_wig)

plt.figure(figsize=(12, 6))
plt.plot(y_test_sip.values, label="Real")
plt.plot(pred_sip, label="Forecast")
plt.legend()
plt.title("XGBoost Forecast vs Real Values for S&P 500")
plt.show()

plt.figure(figsize=(12, 6))
plt.plot(y_test_wig.values, label="Real")
plt.plot(pred_wig, label="Forecast")
plt.legend()
plt.title("XGBoost Forecast vs Real Values for WIG 20")
plt.show()

plot_importance(model_sip)
plt.show()
plot_importance(model_wig)
plt.show()