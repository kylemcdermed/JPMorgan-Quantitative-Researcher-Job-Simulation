import pandas as pd
import numpy as np

def estimate_price(input_date, model, historical_data):
    """
    Estimate natural gas price for any given date using trained Random Forest model.
    
    Parameters:
    - input_date: str or pd.Timestamp, e.g. '2025-10-31'
    - model: trained RandomForestRegressor
    - historical_data: DataFrame containing historical prices + features
    
    Returns:
    - Predicted price (float)
    """
    
    # Convert input to Timestamp
    input_date = pd.to_datetime(input_date)
    
    # Get last historical values
    last_index = historical_data['TimeIndex'].iloc[-1]
    last_price = historical_data['Price'].iloc[-1]
    last_ma3 = historical_data['Price_MA3'].iloc[-1]
    
    # If date is in historical range, predict using features from historical data
    if input_date <= historical_data.index[-1]:
        # Find the closest date in historical data
        closest_row = historical_data.loc[historical_data.index == input_date]
        if not closest_row.empty:
            return float(closest_row['Price'].iloc[0])
        else:
            # Create features from historical trend/seasonality
            month = input_date.month
            time_index = historical_data.index.get_loc(input_date) if input_date in historical_data.index else last_index + 1
            lag1 = historical_data['Price'].iloc[-1]
            ma3 = historical_data['Price_MA3'].iloc[-1]
            X_pred = pd.DataFrame({
                'TimeIndex': [time_index],
                'Month': [month],
                'Price_Lag1': [lag1],
                'Price_MA3': [ma3]
            })
            return float(model.predict(X_pred)[0])
    
    # If date is in future
    else:
        # Generate all months from last historical date up to input_date
        future_dates = pd.date_range(start=historical_data.index[-1] + pd.Timedelta(days=30),
                                     end=input_date, freq='ME')
        future_time_index = np.arange(last_index + 1, last_index + 1 + len(future_dates))
        future_months = future_dates.month
        
        # Prepare DataFrame
        future_df = pd.DataFrame({
            'TimeIndex': future_time_index,
            'Month': future_months,
            'Price_Lag1': last_price,
            'Price_MA3': last_ma3
        }, index=future_dates)
        
        # Iteratively predict
        predicted_prices = []
        for i in range(len(future_df)):
            row = future_df.iloc[i:i+1]
            price_pred = model.predict(row[ ['TimeIndex','Month','Price_Lag1','Price_MA3'] ])[0]
            predicted_prices.append(price_pred)
            
            if i + 1 < len(future_df):
                future_df.iloc[i+1, future_df.columns.get_loc('Price_Lag1')] = price_pred
                # update MA3 as rolling of last 3 predictions
                if i >= 1:
                    ma3 = np.mean(predicted_prices[-3:])
                else:
                    ma3 = np.mean([last_ma3, price_pred])
                future_df.iloc[i+1, future_df.columns.get_loc('Price_MA3')] = ma3
        
        # Return predicted price for input_date
        return float(predicted_prices[-1])
        

# Historical data = `data` from CSV
# Trained model = `model` from Random Forest

# Predict a historical date
estimate_price('2021-02-28', model, data) 

# Predict a future date
estimate_price('2025-12-31', model, data)  # predicted price for end of 2025

