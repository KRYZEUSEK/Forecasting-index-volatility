# Runner script to execute all methods in sequence
import warnings

warnings.filterwarnings('ignore')

print('Running ARIMA...')
from ARIMA import run_arima

run_arima()

print('\nRunning GARCH...')
from GARCH import run_garch

run_garch()

print('\nRunning NeuralNetwork (5 epochs)...')
from NeuralNetwork import run_neural_network

run_neural_network(epochs=5)

print('\nRunning XGBoost...')
from XGBoost import run_xgboost

run_xgboost()

print('\nAll done.')
