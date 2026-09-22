import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score

from pathlib import Path

def train_and_evaluate_model():
    # Load dataset
    root_dir = Path(__file__).parent
    data = pd.read_csv(root_dir / 'data/healthcare-dataset-stroke-data.csv')

    # 2. Separate data into Features (X) and Target (y)
    # X represents the data the model learns from, y is what you want to predict
    X = data[['gender', 'age', 'hypertension', 'heart_disease', 'ever_married', 'work_type', 'Residence_type', 'avg_glucose_level', 'bmi', 'smoking_status']]
    y = data['stroke']

    # Map categorical variables to numerical values
    gender_mapping = {'Male': 0, 'Female': 1, 'Other': 2}
    married_mapping = {'No': 0, 'Yes': 1}
    work_type_mapping = {'Private': 0, 'Self-employed': 1, 'Govt_job': 2, 'children': 3, 'Never_worked': 4}
    residence_mapping = {'Urban': 0, 'Rural': 1}
    smoking_mapping = {'never smoked': 0, 'formerly smoked': 1, 'smokes': 2, 'Unknown': 3}

    # Apply mappings to the dataset
    X['gender'] = X['gender'].map(gender_mapping)
    X['ever_married'] = X['ever_married'].map(married_mapping)
    X['work_type'] = X['work_type'].map(work_type_mapping)
    X['Residence_type'] = X['Residence_type'].map(residence_mapping)
    X['smoking_status'] = X['smoking_status'].map(smoking_mapping)

    # 3. Clean the 'bmi' column (convert "N/A" strings to float and fill missing values)
    data['bmi'] = pd.to_numeric(data['bmi'], errors='coerce')
    data['bmi'] = data['bmi'].fillna(data['bmi'].median())


    # 3. Split the data into Training (80%) and Testing (20%) sets
    # This ensures we can test the model on unseen data to check its accuracy
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # 4. Initialize the machine learning model
    model = RandomForestClassifier(criterion='gini', n_estimators=10, random_state=42)

    # 5. Train the model using the .fit() method
    # This is the actual "learning" step where the model adjusts to your data
    model.fit(X_train.values, y_train)

    # 6. Make predictions on the test data
    predictions = model.predict(X_test.values)

    # 7. Evaluate how well the model performed
    model_accuracy = accuracy_score(y_test, predictions)

    # 8. Print the accuracy of the model
    print(f"Model Accuracy: {model_accuracy * 100:.2f}%")

    return model

def prompt_for_input():
    gender = int(input("Enter gender (0 for Male, 1 for Female, 2 for Other): "))
    age = float(input("Enter age: "))
    hypertension = int(input("Enter hypertension (0 for No, 1 for Yes): "))
    heart_disease = int(input("Enter heart disease (0 for No, 1 for Yes): "))
    ever_married = int(input("Ever married (0 for No, 1 for Yes): "))
    work_type = int(input("Enter work type (0 for Private, 1 for Self-employed, 2 for Govt_job, 3 for children, 4 for Never_worked): "))
    residence_type = int(input("Enter residence type (0 for Urban, 1 for Rural): "))
    avg_glucose_level = float(input("Enter average glucose level: "))
    bmi = float(input("Enter BMI: "))
    smoking_status = int(input("Enter smoking status (0 for never smoked, 1 for formerly smoked, 2 for smokes, 3 for Unknown): "))

    return [[gender, age, hypertension, heart_disease, ever_married, work_type, residence_type, avg_glucose_level, bmi, smoking_status]]

def main():
    model = train_and_evaluate_model()
    input = prompt_for_input()
    prediction = model.predict(input)
    print(f"Predicted Stroke Risk: {'Yes' if prediction[0] == 1 else 'No'}")

if __name__ == "__main__":
    main()