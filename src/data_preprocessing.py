import pandas as pd
from sklearn.preprocessing import OneHotEncoder
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import LabelEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score


file_path = "/Users/sunainasram/loan-approval-prediction/DS-Project/loan-approval-prediction/data/raw/loan_approval_dataset.csv"

def load_data(file_path):
    data=pd.read_csv(file_path)
    return data 

def preprocess(data):
    data.columns=data.columns.str.strip()
    #data=data.dropna(inplace=True)
    data=data.drop(columns=['loan_id'])
    data=data.drop(columns=['no_of_dependents'])
    data=data.drop(columns=['education','self_employed'])
    # Clean any leading/trailing spaces in column names
    data.columns = data.columns.str.strip()

    # Initialize LabelEncoder and overwrite 'loan_status' in a single column
    le = LabelEncoder()
    data['loan_status'] = le.fit_transform(data['loan_status'].str.strip())

    # Save the DataFrame to CSV (use data.to_csv, not pd.to_csv)
    data.to_csv(
        '/Users/sunainasram/Downloads/DS-Project/loan-approval-prediction/data/processed/cleaned_data.csv',
        index=False,
    )

    return data




if __name__ == "__main__":
    data = load_data("/Users/sunainasram/loan-approval-prediction/DS-Project/loan-approval-prediction/data/raw/loan_approval_dataset.csv")
    cleaned_data = preprocess(data)

    cleaned_data.to_csv(
        "/Users/sunainasram/loan-approval-prediction/DS-Project/loan-approval-prediction/data/processed/cleaned_data.csv",
        index=False
    )

    print("Data preprocessing completed!")