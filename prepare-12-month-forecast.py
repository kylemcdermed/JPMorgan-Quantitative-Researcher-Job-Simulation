import numpy as np 
import pandas as pd

# Last values 
last_index = data['TimeIndex'].iloc[-1]
last_price = data['Price'].iloc[-1]
last_ma3 = data['Price_MA3'].iloc[-1]

# Generate next 12 months 
future_dates = pd.date_range(start=data.index[-1] + pd.Timedelta(days=30), periods=12, freq='ME')
future_time_index = np.arange(last_index+1, last_index+13)
future_months = future_dates.month

# Create DataFrame
future_df = pd.DataFrame({
    'TimeIndex' : future_time_index,
    'Month' : future_months
}, index=future_dates)

# Initialize lag and MA3 columns
future_df['Price_Lag1'] = last_price
future_df['Price_MA3'] = last_ma3
