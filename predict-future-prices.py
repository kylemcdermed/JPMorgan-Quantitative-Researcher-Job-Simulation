future_prices = []

for i in range(len(future_df)):
    row = future_df.iloc[i:i+1]
    X_pred = row[features]
    price_pred = model.predict(X_pred)[0]

    # Update lag and MA3 for next iteration
    future_prices.append(price_pred)
    if i + 1 < len(future_df):
        future_df.iloc[i+1, future_df.columns.get_loc('Price_Lag1')] = price_pred
        # Update MA3 as rolling mean of last 3 predicted prices 
        if i >= 1:
            ma3 = np.mean(future_prices[-3:])
        else:
            ma3 = np.mean([last_ma3, price_pred])
        future_df.iloc[i+1, future_df.columns.get_loc('Price_MA3')] = ma3

future_df['PredictedPrice'] = future_prices
