import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
import joblib
from io import StringIO
import numpy as np

# Pima Indians Diabetes Dataset (Source: UCI ML Repository)
data = """
6,148,72,35,0,33.6,0.627,50,1
1,85,66,29,0,26.6,0.351,31,0
8,183,64,0,0,23.3,0.672,32,1
1,89,66,23,94,28.1,0.167,21,0
0,137,40,35,168,43.1,2.288,33,1
5,116,74,0,0,25.6,0.201,30,0
3,78,50,32,88,31.0,0.248,26,1
10,115,0,0,0,35.3,0.134,29,0
2,197,70,45,543,30.5,0.158,53,1
8,125,96,0,0,0.0,0.232,54,1
4,110,92,0,0,37.6,0.191,30,0
10,139,80,0,0,27.1,1.441,57,0
10,139,80,0,0,27.1,1.441,57,0
1,101,60,40,0,35.7,0.484,23,0
... (truncated for brevity, typically load the full 768 rows)
"""
# Note: For the full dataset, you should download the CSV file and use pd.read_csv('pima-indians-diabetes.csv')

# Since I can't access a local file, I use StringIO for this example:
df = pd.read_csv(StringIO(data), header=None)

# Naming columns (matches the required API input order)
df.columns = [
    'Pregnancies', 'Glucose', 'BloodPressure', 'SkinThickness',
    'Insulin', 'BMI', 'DiabetesPedigreeFunction', 'Age', 'Outcome'
]

# Replace zero values (physiologically implausible) with the median for Glucose, BP, BMI
for col in ['Glucose', 'BloodPressure', 'BMI']:
    df[col] = df[col].replace(0, df[col].median())

# Define features (X) and target (y)
X = df.drop('Outcome', axis=1)
y = df['Outcome']

# Split data (optional for this simple project, but good practice)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Initialize and train the Scaler (CRUCIAL for Logistic Regression)
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Initialize and train the Logistic Regression Model (AI/ML Concept)
model = LogisticRegression(solver='liblinear', random_state=42)
model.fit(X_train_scaled, y_train)

# Calculate accuracy (for project documentation)
accuracy = model.score(X_test_scaled, y_test)
print(f"Model Training Complete. Accuracy: {accuracy:.2f}")

# Save the trained model and scaler (Backend Integration)
joblib.dump(model, 'model.pkl')
joblib.dump(scaler, 'scaler.pkl')

# Save feature names for the API (for robust coding)
joblib.dump(X.columns.tolist(), 'feature_names.pkl') 

print("\nModel ('model.pkl'), Scaler ('scaler.pkl'), and Feature Names ('feature_names.pkl') saved successfully.")