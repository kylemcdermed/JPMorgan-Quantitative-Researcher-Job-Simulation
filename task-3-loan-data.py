'''

The risk manager has collected data on the loan borrowers. 
The data is in tabular format, with each row providing details of the borrower, 
including their income, total loans outstanding, and a few other metrics. 
There is also a column indicating if the borrower has previously defaulted on a loan. 
You must use this data to build a model that, given details for any loan described above, 
will predict the probability that the borrower will default (also known as PD: the probability of default). 
Use the provided data to train a function that will estimate the probability of default for a borrower. 
Assuming a recovery rate of 10%, this can be used to give the expected loss on a loan.

You should produce a function that can take in the properties of a loan and output the expected loss.
You can explore any technique ranging from a simple regression or a decision tree to something more advanced. 
You can also use multiple methods and provide a comparative analysis.

'''

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score

data_path = r"C:\Users\kylem\Task 3 and 4_Loan_Data.csv"
data = pd.read_csv(data_path)

# Check first 5 rows to see the columns
# print(data.head())

X = data.drop(columns=['default'])
y = data['default']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

model = LogisticRegression(max_iter=1000)
model.fit(X_train,y_train)

y_pred_prob = model.predict_proba(X_test)[:,1]
auc = roc_auc_score(y_test,y_pred_prob)
print(f'ROC-AUC: {auc:.3f}')

def expected_loss(df, loan_amount='loan_amt_outstanding', recovery_rate=0.10):
    df_features = df.drop(columns=['default'])
    pd_probs = model.predict_proba(df_features)[:,1]
    expected_losses = df[loan_amount] * pd_probs * (1 - recovery_rate)
    df_result = df.copy()
    df_result['Probability_of_Default'] = pd_probs
    df_result['Expected_Loss'] = expected_losses
    return df_result

results_df = expected_loss(data, loan_amount='loan_amt_outstanding')

output_path = r"C:\Users\kylem\Task_3_Loan_Data.csv"
results_df.to_csv(output_path, index=False)
# print(f"Credit risk results saved to: {output_path}")

print(results_df.sort_values('Expected_Loss', ascending=False).head())
