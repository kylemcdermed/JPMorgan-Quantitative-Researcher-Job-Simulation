data['Price_MA3'] = data['Price'].rolling(window=3).mean()

plt.figure(figsize=(14,6))
plt.plot(data.index, data['Price'], label='Monthly Price', marker='o')
plt.plot(data.index, data['Price_MA3'], label='3 Month Moving Average', linestyle='--')
plt.title('Natural Gas Prices with 3 Month Moving Average')
plt.xlabel('Date')
plt.ylabel('Price ($)')
plt.grid(True)
plt.show()O

