'''

TASK

You need to create a prototype pricing model that can go through further validation and testing before being put into production. 
Eventually, this model may be the basis for fully automated quoting to clients, but for now, 
the desk will use it with manual oversight to explore options with the client. 

You should write a function that is able to use the data you created previously to price the contract. 
The client may want to choose multiple dates to inject and withdraw a set amount of gas, 
so your approach should generalize the explanation from before. Consider all the cash flows involved in the product.

The input parameters that should be taken into account for pricing are:

Injection dates. 
Withdrawal dates.
The prices at which the commodity can be purchased/sold on those dates.
The rate at which the gas can be injected/withdrawn.
The maximum volume that can be stored.
Storage costs.
Write a function that takes these inputs and gives back the value of the contract. 
You can assume there is no transport delay and that interest rates are zero. Market holidays, weekends, 
and bank holidays need not be accounted for. Test your code by selecting a few sample inputs.

'''

import pandas as pd

def price_storage_contract(
    injection_dates, withdrawal_dates, injection_rate, withdrawal_rate,
    max_volume, storage_cost, price_data
):
    """
    Prototype pricing model for natural gas storage contract.
    
    Parameters:
    - injection_dates: list of str (YYYY-MM-DD) where gas is injected
    - withdrawal_dates: list of str (YYYY-MM-DD) where gas is withdrawn
    - injection_rate: units injected per date
    - withdrawal_rate: units withdrawn per date
    - max_volume: maximum storage capacity
    - storage_cost: cost per unit per day of storage
    - price_data: pandas DataFrame with ['Date', 'PredictedPrice']
    
    Returns:
    - contract_value: total P&L of the contract
    - inventory_profile: pandas DataFrame with inventory over time
    """

    # Convert date strings to datetime
    injection_dates = pd.to_datetime(injection_dates)
    withdrawal_dates = pd.to_datetime(withdrawal_dates)
    price_data['Date'] = pd.to_datetime(price_data['Date'])

    # Track inventory, cash flows
    inventory = 0
    cashflow = 0
    inventory_record = []

    for idx, row in price_data.iterrows():
        date = row['Date']
        price = row['PredictedPrice']

        # Injection
        if date in injection_dates:
            if inventory + injection_rate <= max_volume:
                inventory += injection_rate
                cashflow -= injection_rate * price  # buying gas
            else:
                print(f"Injection blocked on {date}: storage full!")

        # Withdrawal
        if date in withdrawal_dates:
            if inventory - withdrawal_rate >= 0:
                inventory -= withdrawal_rate
                cashflow += withdrawal_rate * price  # selling gas
            else:
                print(f"Withdrawal blocked on {date}: not enough gas!")

        # Storage cost
        cashflow -= inventory * storage_cost

        # Track inventory 
        inventory_record.append((date, inventory, cashflow))

    inventory_profile = pd.DataFrame(inventory_record, columns=["Date", "Inventory", "Cashflow"])
    return cashflow, inventory_profile
