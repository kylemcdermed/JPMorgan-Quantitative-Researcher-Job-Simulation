import matplotlib.pyplot as plt

plt.figure(figsize=(14,6))
plt.plot(data.index, data['Price'], marker='o', linestyle='-')
plt.title('History of Natural Gas Prices (End-of-Month)')
plt.xlabel('Date')
plt.ylabel('Price ($)')
plt.grid(True)
plt.show()
