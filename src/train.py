
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import pandas as pd
import joblib
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.metrics import accuracy_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import confusion_matrix,ConfusionMatrixDisplay
from sklearn.tree import DecisionTreeClassifier,plot_tree


import os

#from src.data_preprocessing import load_data, preprocess_data
# Load processed dataset
df = pd.read_csv("/Users/sunainasram/loan-approval-prediction/DS-Project/loan-approval-prediction/data/processed/cleaned_data.csv")

# Separate features and target
X=df.drop(columns=['loan_status'])
y = df["loan_status"]

X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.2,random_state=30)
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# X_train_scaled = scaler.fit_transform(X_train)
# X_test_scaled = scaler.transform(X_test)


clf=LogisticRegression(max_iter=10000, random_state=30)
clf.fit(X_train_scaled,y_train)


# # Evaluate model
y_pred = clf.predict(X_test_scaled)

accuracy=accuracy_score(y_test,y_pred)
print("**********************************************************************")

print("Logistic regression Accuracy is:",accuracy)
print("logistic regression Classification Report:")
print(classification_report(y_test, y_pred))

# Save model
os.makedirs("models", exist_ok=True)

joblib.dump(clf, "models/loan_model_lr.pkl")

print("Logistic regression Model training completed!")

print("**********************************************************************")

clf_decision = DecisionTreeClassifier(criterion="gini", max_depth=3, random_state=42)
clf_decision.fit(X_train, y_train)
y_pred=clf_decision.predict(X_test)
print(f"Decision tree Accuracy: {accuracy_score(y_test, y_pred)}")

y_pred = clf_decision.predict(X_test)
cm = confusion_matrix(y_test, y_pred)
print("decision tree Confusion Matrix:\n", cm)

# Save model
os.makedirs("models", exist_ok=True)

joblib.dump(clf_decision, "models/loan_model_dt.pkl")

print("Decision tree Model training completed!")
print("**********************************************************************")


#Initialize and train the Random Forest
rf = RandomForestClassifier(
    n_estimators=100,  # Number of trees in the forest
    max_depth=10,  # Limits tree depth to prevent overfitting
    random_state=42,
)
rf.fit(X_train, y_train)

# 3. Check Train vs. Test Accuracy
train_acc = rf.score(X_train, y_train)
test_acc = rf.score(X_test, y_test)

print(f"Train Accuracy: {train_acc:.4f}")
print(f"Test Accuracy:  {test_acc:.4f}")

# 4. Optional: Detailed precision, recall, and F1-score
y_pred = rf.predict(X_test)
print("\nClassification Report:\n", classification_report(y_test, y_pred))


# Save model
os.makedirs("models", exist_ok=True)

joblib.dump(rf, "models/loan_model_rf.pkl")

print("Random forest Model training completed!")