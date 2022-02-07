# Data Preprocessing Tools

# Importing the libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn import metrics

# Importing the dataset
dataset = pd.read_csv('YPFD_Cotizaciones_historicas.csv')
dataset.set_index("fecha", inplace=True)

X = dataset.iloc[:, 0:5].values
y = dataset.iloc[:, 3:4].values

# Taking care of missing data
from sklearn.impute import SimpleImputer
imputer = SimpleImputer(missing_values=np.nan, strategy='mean')
imputer.fit(X)
X = imputer.transform(X)

# Splitting the dataset into the Training set and Test set
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2)
#print(X_train); #print(X_test); #print(y_train);#print(y_test)

# Training the Random Forest Regression model
from sklearn.ensemble import RandomForestRegressor
regressor = RandomForestRegressor(n_estimators = 100)
regressor.fit(X_train, y_train)
print('--------------------------------------')
print(regressor.score(X_test, y_test))
print('--------------------------------------')
predict = regressor.predict(X_test)

print("Mean Absolute Error:", round(metrics.mean_absolute_error(y_test, predict), 4))
print("Mean Squared Error:", round(metrics.mean_squared_error(y_test, predict), 4))
print("Root Mean Squared Error:", round(np.sqrt(metrics.mean_squared_error(y_test, predict)), 4))
print("(R^2) Score:", round(metrics.r2_score(y_test, predict), 4))
print(f'Train Score : {regressor.score(X_train, y_train) * 100:.2f}% and Test Score : {regressor.score(X_test, y_test) * 100:.2f}% using Random Tree Regressor.')
errors = abs(predict - y_test)
mape = 100 * (errors / y_test)
accuracy = 100 - np.mean(mape)
print('Accuracy:', round(accuracy, 2), '%.')

predictions = pd.DataFrame({"Predictions": predict}, 
                           index=pd.date_range(start=dataset.index[-1],
                                               periods=len(predict), freq="D"))
predictions.to_csv("Predicted-price-data.csv")

# predicciones a un mes
onemonth_df = pd.DataFrame(predictions[:21])
onemonth_df.to_csv("one-month-predictions.csv")

#Predicciones a un año
oneyear_df = pd.DataFrame(predictions[:252])
oneyear_df.to_csv("one-year-predictions.csv")


# plot predicciones a un mes
onemonth_df_pred = pd.read_csv("one-month-predictions.csv")
onemonth_df_pred.columns = ['Date', 'Predictions']
onemonth_df_pred.set_index("Date", inplace=True)

buy_price = min(onemonth_df_pred["Predictions"])
sell_price = max(onemonth_df_pred["Predictions"])
onemonth_buy = onemonth_df_pred.loc[onemonth_df_pred["Predictions"] == buy_price]
onemonth_sell = onemonth_df_pred.loc[onemonth_df_pred["Predictions"] == sell_price]
print("Buy price and date")
print(onemonth_buy)
print("Sell price and date")
print(onemonth_sell)
#onemonth_df_pred["Predictions"].plot(figsize=(10, 5), title="Forecast for the next 1 month", color="blue")
#plt.xlabel("Date")
#plt.ylabel("Price")
#plt.legend()

"""
# Prediccion a un año

oneyear_df_pred = pd.read_csv("one-year-predictions.csv")
oneyear_df_pred.columns = ['Fecha', 'Predictions']
oneyear_df_pred.set_index("Fecha", inplace=True)
buy_price = min(oneyear_df_pred["Predictions"])
sell_price = max(oneyear_df_pred["Predictions"])
oneyear_buy = oneyear_df_pred.loc[oneyear_df_pred["Predictions"] == buy_price]
oneyear_sell = oneyear_df_pred.loc[oneyear_df_pred["Predictions"] == sell_price]
print("Buy price and date")
print(oneyear_buy)
print("Sell price and date")
print(oneyear_sell)
oneyear_df_pred["Predictions"].plot(figsize=(10, 5), title="Forecast for the next 1 year", color="blue")
plt.xlabel("Date")
plt.ylabel("Price")
plt.legend()
"""






