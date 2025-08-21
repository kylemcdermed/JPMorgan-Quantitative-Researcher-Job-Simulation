import seaborn as sns

data['Month'] = data.index.month

plt.figure(figsize=(12,6))
sns.boxplot(x='Month', y='Price', data=data)
plt.title('Natural Gas Prices (Seasonality Check)')
plt.xlabel('Month')
plt.ylabel('Price ($)')
# plt.grid(True)
plt.show()
