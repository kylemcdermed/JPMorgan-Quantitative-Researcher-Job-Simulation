# Storage Contract Pricing Function
def price_storage_contract(injection_dates, withdrawal_dates, rate, max_volume, storage_cost_rate, price_func):
    """
    Prices a natural gas storage contract based on given parameters.
    
    Parameters:
    - injection_dates: list of str or datetime, dates for possible injection
    - withdrawal_dates: list of str or datetime, dates for possible withdrawal
    - rate: float, max injection/withdrawal rate per date (assumed per operation)
    - max_volume: float, maximum storage capacity
    - storage_cost_rate: float, storage cost per unit volume per day
    - price_func: callable, function that takes a datetime and returns the price on that date
    
    Returns:
    - float, the maximum value (profit) of the contract
    """
    # Convert dates to datetime
    injection_dates = [pd.to_datetime(d) for d in injection_dates]
    withdrawal_dates = [pd.to_datetime(d) for d in withdrawal_dates]
    
    # Get all unique sorted dates
    all_dates = sorted(set(injection_dates + withdrawal_dates))
    
    if not all_dates:
        return 0.0
    
    # Get prices for each date
    prices = {d: price_func(d) for d in all_dates}
    
    # Set up the LP problem
    prob = pulp.LpProblem("Natural_Gas_Storage_Contract_Value", pulp.LpMaximize)
    
    # Variables: injection and withdrawal amounts
    inject_vars = {d: pulp.LpVariable(f"inject_{d.date()}", lowBound=0, upBound=rate) for d in injection_dates}
    withdraw_vars = {d: pulp.LpVariable(f"withdraw_{d.date()}", lowBound=0, upBound=rate) for d in withdrawal_dates}
    
    # Inventory after each date
    inventory_vars = {d: pulp.LpVariable(f"inv_{d.date()}", lowBound=0, upBound=max_volume) for d in all_dates}
    
    # Objective: Maximize (revenue from sales - cost of purchases - storage costs)
    revenue = pulp.lpSum([withdraw_vars[d] * prices[d] for d in withdrawal_dates])
    purchase_cost = pulp.lpSum([inject_vars[d] * prices[d] for d in injection_dates])
    
    storage_cost = 0
    for i in range(len(all_dates) - 1):
        d_current = all_dates[i]
        d_next = all_dates[i + 1]
        delta_days = (d_next - d_current).days
        storage_cost += storage_cost_rate * inventory_vars[d_current] * delta_days
    
    prob += revenue - purchase_cost - storage_cost
    
    # Constraints: Inventory balance
    for i, d in enumerate(all_dates):
        net_flow = 0
        if d in injection_dates:
            net_flow += inject_vars[d]
        if d in withdrawal_dates:
            net_flow -= withdraw_vars[d]
        
        if i == 0:
            prob += inventory_vars[d] == net_flow
        else:
            prev_d = all_dates[i - 1]
            prob += inventory_vars[d] == inventory_vars[prev_d] + net_flow
    
    # Solve the problem 
    prob.solve(pulp.PULP_CBC_CMD(msg=False))
    
    # Check if optimal solution found
    if prob.status != pulp.LpStatusOptimal:
        print("No optimal solution found.")
        return 0.0
    
    return pulp.value(prob.objective)


# Test the Pricing Function
# Example 1: Simple case with low summer prices for injection, high winter for withdrawal
injection_dates = ['2024-06-30', '2024-07-31']
withdrawal_dates = ['2024-12-31', '2025-01-31']
rate = 1000000  # e.g., 1 million units per operation
max_volume = 2000000  # max storage 2 million units
storage_cost_rate = 0.0001  # $0.0001 per unit per day
price_func = lambda date: estimate_price(date, model, data)

value1 = price_storage_contract(injection_dates, withdrawal_dates, rate, max_volume, storage_cost_rate, price_func)
print(f"Contract Value: ${value1:,.2f}")

# Example 2: More dates, injecting in spring/summer, withdrawing in fall/winter
injection_dates2 = ['2024-04-30', '2024-05-31', '2024-06-30']
withdrawal_dates2 = ['2024-11-30', '2024-12-31', '2025-01-31']
rate2 = 700000  # lower rate
max_volume2 = 2000000
storage_cost_rate2 = 0.00005  # lower cost rate

value2 = price_storage_contract(injection_dates2, withdrawal_dates2, rate2, max_volume2, storage_cost_rate2, price_func)
# print(f"Contract Value: ${value2:,.2f}")

# Example 3: Single injection and withdrawal
injection_dates3 = ['2024-07-31']
withdrawal_dates3 = ['2024-12-31']
rate3 = 1000000
max_volume3 = 1000000
storage_cost_rate3 = 0.0001

value3 = price_storage_contract(injection_dates3, withdrawal_dates3, rate3, max_volume3, storage_cost_rate3, price_func)
# print(f"Contract Value: ${value3:,.2f}")
