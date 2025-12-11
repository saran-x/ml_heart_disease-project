from flask import Flask, render_template, request
import joblib
import pandas as pd

app = Flask(__name__)

# Load model
model = joblib.load("../model/Heart_disease_model.pkl")
train_column = joblib.load("../model/train_columns.pkl")

# Clean user input to match training columns
def process_input(user_input):
    df = pd.DataFrame([user_input])

    # Missing columns → add 0
    miss = set(train_column) - set(df.columns)
    for col in miss:
        df[col] = 0

    # Extra columns → drop
    extra = set(df.columns) - set(train_column)
    df.drop(columns=list(extra), inplace=True)

    # Arrange order
    df = df[train_column]
    return df


@app.route("/")
def home():
    return render_template("index.html")


@app.route("/predict", methods=["POST"])
def predict():
    if request.method == "POST":

        # Numeric fields
        BMI = float(request.form["BMI"])
        PhysicalHealth = float(request.form["PhysicalHealth"])
        MentalHealth = float(request.form["MentalHealth"])
        SleepTime = float(request.form["SleepTime"])

        # Binary fields
        Smoking = int(request.form["Smoking"])
        AlcoholDrinking = int(request.form["AlcoholDrinking"])
        Stroke = int(request.form["Stroke"])
        DiffWalking = int(request.form["DiffWalking"])
        Sex = int(request.form["Sex"])
        PhysicalActivity = int(request.form["PhysicalActivity"])
        Asthma = int(request.form["Asthma"])
        KidneyDisease = int(request.form["KidneyDisease"])
        SkinCancer = int(request.form["SkinCancer"])

        # Age map
        age_map = {
            '18-24': 21, '25-29': 27, '30-34': 32, '35-39': 37,
            '40-44': 42, '45-49': 47, '50-54': 52, '55-59': 57,
            '60-64': 62, '65-69': 67, '70-74': 72, '75-79': 77,
            '80 or older': 82
        }
        AgeCategory = age_map[request.form["AgeCategory"]]

        GenHealth = int(request.form["GenHealth"])

        # Diabetic dummy
        d = request.form["Diabetic"]
        diabetic_dummy = {
            'Diabetic_No borderline diabetes': 0,
            'Diabetic_Yes': 0,
            'Diabetic_Yes (during pregnancy)': 0
        }
        diabetic_dummy[d] = 1

        # Race dummy
        r = request.form["Race"]
        race_dummy = {
            'Race_Asian': 0,
            'Race_Black': 0,
            'Race_Hispanic': 0,
            'Race_Other': 0,
            'Race_White': 0
        }
        race_dummy[r] = 1

        # Final input
        input_data = {
            'BMI': BMI,
            'Smoking': Smoking,
            'AlcoholDrinking': AlcoholDrinking,
            'Stroke': Stroke,
            'PhysicalHealth': PhysicalHealth,
            'MentalHealth': MentalHealth,
            'DiffWalking': DiffWalking,
            'Sex': Sex,
            'AgeCategory': AgeCategory,
            'PhysicalActivity': PhysicalActivity,
            'GenHealth': GenHealth,
            'SleepTime': SleepTime,
            'Asthma': Asthma,
            'KidneyDisease': KidneyDisease,
            'SkinCancer': SkinCancer,
            'Diabetic_No, borderline diabetes': diabetic_dummy['Diabetic_No borderline diabetes'],
            'Diabetic_Yes': diabetic_dummy['Diabetic_Yes'],
            'Diabetic_Yes (during pregnancy)': diabetic_dummy['Diabetic_Yes (during pregnancy)'],
            'Race_Asian': race_dummy['Race_Asian'],
            'Race_Black': race_dummy['Race_Black'],
            'Race_Hispanic': race_dummy['Race_Hispanic'],
            'Race_Other': race_dummy['Race_Other'],
            'Race_White': race_dummy['Race_White']
        }

        df = process_input(input_data)
        print(df)
        prediction = model.predict(df)[0]
        result = "💔 Patient Has Heart Disease" if prediction == 1 else "💓 Patient Does NOT Have Heart Disease"

        return render_template("result.html", results=result)

    return render_template("result.html", results="Error: Invalid request")


if __name__ == "__main__":
    app.run(debug=True)
