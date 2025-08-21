from sklearn.ensemble import RandomForestRegressor

# Features: trend + seasonality + lag/rolling MA 
features = ['TimeIndex', 'Month', 'Price_Lag1', 'Price_MA3']
X = data[features]
y = data['Price']

# Initialize model
model = RandomForestRegressor(n_estimators=200, random_state=42)

# Fit model
model.fit(X,y)

# Check training performance
train_preds = model.predict(X)

import matplotlib.pyplot as plt

plt.figure(figsize=(12,6))
plt.plot(data.index, y, label='Actual Price', marker='o')
plt.plot(data.index, train_preds, label='Predicted Price', marker='x', linestyle='--')
plt.title('Random Forest Fit on Historical Natural Gas Prices')
plt.xlabel('Date')
plt.ylabel('Price ($)')
plt.legend()
plt.grid(True)
plt.show()
