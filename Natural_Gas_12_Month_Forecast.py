# Import Libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestRegressor


# Grab Data
data = pd.read_csv('Nat_Gas.csv')
print(data.columns)  # Debug: Check column names
print(data['Dates'].head())  # Debug: Check date format
data.rename(columns={'Prices': 'Price'}, inplace=True)  # Adjust column name if needed
data['Dates'] = pd.to_datetime(data['Dates'], format='%m/%d/%y', errors='raise')  # Adjust format as needed
data.set_index('Dates', inplace=True)


# Visualize Historical Data
plt.figure(figsize=(12,6))
plt.plot(data.index, data['Price'], label='Historical Price', marker='o')
plt.title('Natural Gas Historical Prices (Monthly)')
plt.xlabel('Date')
plt.ylabel('Price ($)')
plt.grid(True)
plt.legend()
plt.show()

# Monthly boxplot for seasonality
data['Month'] = data.index.month
plt.figure(figsize=(12,6))
data.boxplot(column='Price', by='Month')
plt.title('Monthly Price Variation')
plt.xlabel('Month')
plt.ylabel('Price ($)')
plt.suptitle('')
plt.show()


# Feature Engineering
data['TimeIndex'] = np.arange(len(data))  # sequential trend
data['Price_Lag1'] = data['Price'].shift(1).bfill()
data['Price_MA3'] = data['Price'].rolling(3, min_periods=1).mean()

features = ['TimeIndex', 'Month', 'Price_Lag1', 'Price_MA3']
X = data[features]
y = data['Price']


# Train Random Forest Model
model = RandomForestRegressor(n_estimators=200, random_state=42)
model.fit(X, y)

# Visual check of fit
train_preds = model.predict(X)
plt.figure(figsize=(12,6))
plt.plot(data.index, y, label='Actual Price', marker='o')
plt.plot(data.index, train_preds, label='Predicted Price', marker='x', linestyle='--')
plt.title('Random Forest Fit on Historical Natural Gas Prices')
plt.xlabel('Date')
plt.ylabel('Price ($)')
plt.legend()
plt.grid(True)
plt.show()


# Forecast Next 12 Months
last_index = data['TimeIndex'].iloc[-1]
last_price = data['Price'].iloc[-1]
last_ma3 = data['Price_MA3'].iloc[-1]

future_dates = pd.date_range(start=data.index[-1] + pd.Timedelta(days=30),
                             periods=12, freq='ME')
future_time_index = np.arange(last_index+1, last_index+13)
future_months = future_dates.month

future_df = pd.DataFrame({
    'TimeIndex': future_time_index,
    'Month': future_months,
    'Price_Lag1': last_price,
    'Price_MA3': last_ma3
}, index=future_dates)

future_prices = []

for i in range(len(future_df)):
    row = future_df.iloc[i:i+1]  # keep as DataFrame
    price_pred = model.predict(row[features])[0]
    future_prices.append(price_pred)
    
    if i+1 < len(future_df):
        future_df.iloc[i+1, future_df.columns.get_loc('Price_Lag1')] = price_pred
        if i >= 1:
            ma3 = np.mean(future_prices[-3:])
        else:
            ma3 = np.mean([last_ma3, price_pred])
        future_df.iloc[i+1, future_df.columns.get_loc('Price_MA3')] = ma3

future_df['PredictedPrice'] = future_prices

# Visualize Forecast
plt.figure(figsize=(12,6))
plt.plot(data.index, data['Price'], label='Historical', marker='o')
plt.plot(future_df.index, future_df['PredictedPrice'], label='Forecast (12 months)', marker='x', linestyle='--')
plt.title('Natural Gas Prices: Historical + Forecast')
plt.xlabel('Date')
plt.ylabel('Price ($)')
plt.legend()
plt.grid(True)
plt.show()


# Forecast Table for Next 12 Months
forecast_table = pd.DataFrame({
    'Date': future_df.index,
    'PredictedPrice': future_df['PredictedPrice'].round(2)
})
forecast_table.reset_index(drop=True, inplace=True)

# print("12-Month Natural Gas Price Forecast:")
# print(forecast_table)

# Optional: save table to CSV
forecast_table.to_csv('Natural_Gas_12_Month_Forecast.csv', index=False)


# Price Lookup Function
def estimate_price(input_date, model, historical_data):
    input_date = pd.to_datetime(input_date)
    
    # Last historical values
    last_index = historical_data['TimeIndex'].iloc[-1]
    last_price = historical_data['Price'].iloc[-1]
    last_ma3 = historical_data['Price_MA3'].iloc[-1]
    
    if input_date <= historical_data.index[-1]:
        closest_row = historical_data.loc[historical_data.index == input_date]
        if not closest_row.empty:
            return float(closest_row['Price'].iloc[0])
        else:
            month = input_date.month
            time_index = historical_data.index.get_loc(input_date) if input_date in historical_data.index else last_index+1
            lag1 = historical_data['Price'].iloc[-1]
            ma3 = historical_data['Price_MA3'].iloc[-1]
            X_pred = pd.DataFrame({
                'TimeIndex': [time_index],
                'Month': [month],
                'Price_Lag1': [lag1],
                'Price_MA3': [ma3]
            })
            return float(model.predict(X_pred)[0])
    else:
        future_dates = pd.date_range(start=historical_data.index[-1] + pd.Timedelta(days=30),
                                     end=input_date, freq='ME')
        future_time_index = np.arange(last_index+1, last_index+1+len(future_dates))
        future_months = future_dates.month
        
        future_df = pd.DataFrame({
            'TimeIndex': future_time_index,
            'Month': future_months,
            'Price_Lag1': last_price,
            'Price_MA3': last_ma3
        }, index=future_dates)
        
        predicted_prices = []
        for i in range(len(future_df)):
            row = future_df.iloc[i:i+1]
            price_pred = model.predict(row[features])[0]
            predicted_prices.append(price_pred)
            if i+1 < len(future_df):
                future_df.iloc[i+1, future_df.columns.get_loc('Price_Lag1')] = price_pred
                if i >= 1:
                    ma3 = np.mean(predicted_prices[-3:])
                else:
                    ma3 = np.mean([last_ma3, price_pred])
                future_df.iloc[i+1, future_df.columns.get_loc('Price_MA3')] = ma3
        
        return float(predicted_prices[-1])


# Display 12-Month Forecast
print("12-Month Natural Gas Price Forecast (Real Predictions):")
print(forecast_table)
