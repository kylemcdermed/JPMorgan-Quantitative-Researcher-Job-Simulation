# This data is available for roughly the next 18 months and is combined with historical prices in a time series database
# You should use this monthly snapshot to produce a varying picture of the existing price data, as well as an extrapolation for an extra year, 
# in case the client needs an indicative price for a longer-term storage contract.
# Try to visualize the data to find patterns and consider what factors might cause the price of natural gas to vary. 
# This can include looking at months of the year for seasonal trends that affect the prices, but market holidays, weekends, and bank holidays 
# need not be accounted for.


# Download the monthly natural gas price data.
# Each point in the data set corresponds to the purchase price of natural gas at the end of a month, from 31st October 2020 to 30th September 2024.
# Analyze the data to estimate the purchase price of gas at any date in the past and extrapolate it for one year into the future. 
# Your code should take a date as input and return a price estimate.

import os
import shutil
import pandas as pd


# Grab --> Date and Price from Nat_Gas.csv
data = pd.read_csv('Nat_Gas.csv', parse_dates=['Dates'])
data.set_index('Dates', inplace=True)
data.rename(columns={'Prices' : 'Price'}, inplace=True)

print(data.head())O

