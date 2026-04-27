import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, mean_squared_error

from scipy.stats import ttest_ind

# STYLE SETTINGS
sns.set_style("whitegrid")
sns.set_palette("pastel")
plt.rcParams['figure.figsize'] = (8,5)

# LOAD DATASET
df = pd.read_csv('Indian Personal Finance and Spending Habits.csv')

# REMOVE IRRELEVANT COLUMNS
columns_to_drop = [
    'Desired_Savings',
    'Disposable_Income',
    'Potential_Savings_Groceries',
    'Potential_Savings_Transport',
    'Potential_Savings_Eating_Out',
    'Potential_Savings_Entertainment',
    'Potential_Savings_Utilities',
    'Potential_Savings_Healthcare',
    'Potential_Savings_Education',
    'Potential_Savings_Miscellaneous'
]
df = df.drop(columns=columns_to_drop, errors='ignore')

# HANDLE MISSING VALUES
df = df.dropna()

# MADE THE DATASET OF ~8000 ROWS
if len(df) >= 8000:
    df = df.sample(n=8000, random_state=42)
else:
    df = pd.concat([df]*5, ignore_index=True)
    df = df.sample(n=8000, random_state=42)

# FEATURE ENGINEERING
df['Total_Expenses'] = (
    df['Rent'] + df['Loan_Repayment'] + df['Insurance'] +
    df['Groceries'] + df['Transport'] + df['Eating_Out'] +
    df['Entertainment'] + df['Utilities'] + df['Healthcare'] +
    df['Education'] + df['Miscellaneous']
)

df['Savings'] = df['Income'] - df['Total_Expenses']
df['Savings_Rate'] = df['Savings'] / df['Income']
df['Expense_Ratio'] = df['Total_Expenses'] / df['Income']

# SAVE CLEAN DATASET
df.to_csv('cleaned_finance_dataset.csv', index=False)


# OUTPUT CHECK
print("Final Shape:", df.shape)
print("\nColumns:\n", df.columns)
print(df.head())

# VISUALIZATION STARTS HERE

# 1. Income Distribution (Outlier handled)
plt.figure()
sns.histplot(
    df[df['Income'] < df['Income'].quantile(0.95)]['Income'],
    bins=30, kde=True, color='skyblue'
)
plt.title("Income Distribution")
plt.xlabel("Income")
plt.ylabel("Frequency")
plt.show()

# 2. Income vs Savings (Cleaned)
plt.figure()
sns.scatterplot(
    data=df[df['Income'] < df['Income'].quantile(0.95)],
    x='Income', y='Savings',
    color='mediumseagreen'
)
plt.title("Income vs Savings")
plt.xlabel("Income")
plt.ylabel("Savings")
plt.show()

# 3. Savings Rate Distribution
plt.figure()
sns.histplot(df['Savings_Rate'], bins=30, color='orchid')
plt.title("Savings Rate Distribution")
plt.xlabel("Savings Rate")
plt.ylabel("Count")
plt.show()

# 4. Expense Ratio Distribution
plt.figure()
sns.histplot(df['Expense_Ratio'], bins=30, color='gold')
plt.title("Expense Ratio Distribution")
plt.xlabel("Expense Ratio")
plt.ylabel("Count")
plt.show()

# 5. Expense Ratio by City Tier
plt.figure()
sns.boxplot(
    x='City_Tier',
    y='Expense_Ratio',
    hue='City_Tier',
    data=df,
    palette='pastel',
    legend=False
)
plt.title("Expense Ratio by City Tier")
plt.xlabel("City Tier")
plt.ylabel("Expense Ratio")
plt.show()

# 6. Age vs Savings (with trend line)
plt.figure()
sns.regplot(
    x='Age',
    y='Savings',
    data=df,
    scatter_kws={'alpha':0.5},
    color='teal'
)
plt.title("Age vs Savings (Trend)")
plt.xlabel("Age")
plt.ylabel("Savings")
plt.show()


# 7. Category-wise Spending
expense_cols = [
    'Rent','Loan_Repayment','Insurance','Groceries','Transport',
    'Eating_Out','Entertainment','Utilities','Healthcare','Education','Miscellaneous'
]

avg_expense = df[expense_cols].mean()

plt.figure()
avg_expense.plot(kind='bar', color='cornflowerblue')
plt.title("Average Spending by Category")
plt.xlabel("Expense Category")
plt.ylabel("Average Amount")
plt.xticks(rotation=45)
plt.show()

# 8. Spending Percentage (Pie Chart)
total_avg = df[expense_cols].sum()
percent = (total_avg / total_avg.sum()) * 100

plt.figure()
percent.plot(kind='pie', autopct='%1.1f%%')
plt.title("Spending Percentage Distribution")
plt.ylabel("")
plt.show()

# 9. Saver Category
df['Saver_Type'] = pd.cut(
    df['Savings_Rate'],
    bins=[-1,0.1,0.3,1],
    labels=['Low Saver','Medium Saver','High Saver']
)

plt.figure()
sns.countplot(x='Saver_Type', data=df)
plt.title("Saver Category Distribution")
plt.xlabel("Saver Type")
plt.ylabel("Count")
plt.show()

# 10. Clean Correlation Heatmap
important_cols = [
    'Income','Total_Expenses','Savings',
    'Savings_Rate','Expense_Ratio'
]

plt.figure()
sns.heatmap(df[important_cols].corr(), annot=True, cmap='coolwarm')
plt.title("Key Feature Correlation")
plt.show()


# SIMPLE LINEAR REGRESSION
X = df[['Income']]   # independent variable
y = df['Savings']    # dependent variable

# Split data
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Train model
model = LinearRegression()
model.fit(X_train, y_train)

# Predictions
y_pred = model.predict(X_test)  

# Results
print("\n--- Linear Regression ---")
print("Coefficient (Slope):", model.coef_[0])
print("Intercept:", model.intercept_)
print("R2 Score:", r2_score(y_test, y_pred))
print("MSE:", mean_squared_error(y_test, y_pred))


# Filter data
tier1 = df[df['City_Tier'] == 'Tier_1']['Savings'].dropna()
tier2 = df[df['City_Tier'] == 'Tier_2']['Savings'].dropna()

# Perform t-test
t_stat, p_value = ttest_ind(tier1, tier2)

print("\n--- T-Test (Tier1 vs Tier2 Savings) ---")
print("t-statistic:", t_stat)
print("p-value:", p_value)

# Interpretation
if p_value < 0.05:
    print("Result: Significant difference")
else:
    print("Result: No significant difference")
