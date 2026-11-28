import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns


data=pd.read_csv('/Users/sunainasram/loan-approval-prediction/DS-Project/loan-approval-prediction/data/raw/loan_approval_dataset.csv')

print(data.head())
print(data.info())
print(data.describe())

data.columns=data.columns.str.strip()
corr_matrix = data.corr(numeric_only=True)
print(corr_matrix)
print(data['loan_status'].value_counts())
print(data['loan_status'].value_counts(normalize=True))
print(data.sample(4))
print(data.describe(include=object))

data.loc[3812]
print(data.loc[[3813,3836]])
print(data.loc[[0,1]])

print(data.iloc[0:2,0:3]) 
print(data.iloc[::2])
print(data['loan_status'].unique()) 
data['loan_status']=data['loan_status'].str.strip()
print(data['loan_status'])
print(data.loc[(data['loan_status']=='Approved') & (data['cibil_score']<500)])

plt.scatter(data['income_annum'],data['bank_asset_value'])
plt.title("Scatter plot between income annual and bank asset value")
plt.savefig("/Users/sunainasram/loan-approval-prediction/DS-Project/loan-approval-prediction/reports/scatterplot.png")
plt.show()


sns.pairplot(data)
plt.title("PAIRPLOT")
plt.savefig("/Users/sunainasram/loan-approval-prediction/DS-Project/loan-approval-prediction/reports/pairplot.png")

plt.show()

plt.hist(data['loan_status'])
plt.title("Histogram of loan status")
plt.savefig("/Users/sunainasram/loan-approval-prediction/DS-Project/loan-approval-prediction/reports/histogram.png")

plt.show()

sns.displot(data['loan_amount'],kind='kde')
plt.title("Displot")
plt.savefig("/Users/sunainasram/loan-approval-prediction/DS-Project/loan-approval-prediction/reports/displot.png")

plt.show()


sns.histplot(x=data['bank_asset_value'])
plt.title("Hist plot bank asset value")
plt.savefig("/Users/sunainasram/loan-approval-prediction/DS-Project/loan-approval-prediction/reports/histplot.png")

plt.show()

sns.boxplot(data['loan_amount'])
plt.title("Boxplot of loan amount")
plt.savefig("/Users/sunainasram/loan-approval-prediction/DS-Project/loan-approval-prediction/reports/boxplot.png")

plt.show()

sns.boxplot(data['income_annum'])
plt.title("Boxplot of annual income")

plt.savefig("/Users/sunainasram/loan-approval-prediction/DS-Project/loan-approval-prediction/reports/boxplot.png")

plt.show()

sns.boxplot(x=data['cibil_score'])
plt.title("Boxplot of cibil score")

plt.savefig("/Users/sunainasram/loan-approval-prediction/DS-Project/loan-approval-prediction/reports/boxplot.png")

plt.show()



print(data.isnull().sum())
print(data.isna().sum())



      