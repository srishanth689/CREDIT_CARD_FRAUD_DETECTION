import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report

train_df = pd.read_csv("fraudTrain.csv").head(10)
test_df  = pd.read_csv("fraudTest.csv").head(10)
print("Train shape:", train_df.shape)
print("Test shape :", test_df.shape)
drop_cols = [
    'Unnamed: 0', 'trans_date_trans_time', 'cc_num',
    'merchant', 'first', 'last', 'street',
    'city', 'state', 'zip', 'job', 'dob', 'trans_num'
]
train_df.drop(columns=drop_cols, inplace=True, errors='ignore')
test_df.drop(columns=drop_cols, inplace=True, errors='ignore')
X_train = train_df.drop('is_fraud', axis=1)
y_train = train_df['is_fraud']
X_test = test_df.drop('is_fraud', axis=1)
y_test = test_df['is_fraud']
X_train = pd.get_dummies(X_train)
X_test  = pd.get_dummies(X_test)
X_train, X_test = X_train.align(X_test, join='left', axis=1, fill_value=0)
model = RandomForestClassifier(n_estimators=10, random_state=42)
model.fit(X_train, y_train)
y_pred = model.predict(X_test)
print("\nAccuracy:", accuracy_score(y_test, y_pred))
print("\nClassification Report:\n")
print(classification_report(y_test, y_pred, zero_division=0))