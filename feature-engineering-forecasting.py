data['TimeIndex'] = range(len(data))

data['Month'] = data.index.month
month_dummies = pd.get_dummies(data['Month'], prefix='Month', drop_first=True)
data = pd.concat([data, month_dummies], axis=1)

data['Price_Lag1'] = data['Price'].shift(1)
data.dropna(inplace=True)
