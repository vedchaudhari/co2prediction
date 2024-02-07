import pandas as pd
from sklearn.linear_model import LinearRegression
import pickle

df = pd.read_csv('Fueldata.csv')

# Assuming 'CO2EMISSIONS' is the target variable
X = df[['ENGINESIZE', 'CYLINDERS', 'FUELCONSUMPTION_COMB']]
y = df['CO2EMISSIONS']

regressor = LinearRegression()

# Fitting
regressor.fit(X, y)

# Serialize model
pickle.dump(regressor, open('model.pkl', 'wb'))

# Load model
loaded_model = pickle.load(open('model.pkl', 'rb'))

# Correct input format for prediction (using DataFrame)
prediction_input = pd.DataFrame([[2.6, 8, 10.1]], columns=['ENGINESIZE', 'CYLINDERS', 'FUELCONSUMPTION_COMB'])


# Use the loaded model for prediction
prediction_result = loaded_model.predict(prediction_input)

print(prediction_result)
