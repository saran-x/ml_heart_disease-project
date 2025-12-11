----------------------------------------
Heart Disease Prediction Web App
----------------------------------------
------------------------------------------------------------------------------------------------------------------------------------------
Project Overview

The Heart Disease Prediction Web App is a machine learning project that predicts whether a patient is at risk of heart disease based on health and lifestyle parameters. This project combines data preprocessing, machine learning modeling, and a Flask web application with a Bootstrap front-end to provide a user-friendly interface.
------------------------------------------------------------------------------------------------------------------------------------------

------------------------------------------------------------------------------------------------------------------------------------------

"Features"

 1 .Predicts Heart Disease risk as YES or NO.

 2.Inputs include:

    ==>BMI, Physical Health, Mental Health, Sleep Time
    ==>Smoking, Alcohol Drinking, Stroke, Difficulty Walking
    ==>Sex, Age Category, Physical Activity, General Health
    ==>Asthma, Kidney Disease, Skin Cancer, Diabetes,Race
 3.Bootstrap UI with dynamic results display (red 💔 if at risk, green 💓 if not).
 4.Handles imbalanced dataset using oversampling techniques.
 5.reprocessing ensures all input features match the trained model.
------------------------------------------------------------------------------------------------------------------------------------------

------------------------------------------------------------------------------------------------------------------------------------------
"Technologies Used"

1.Python – Data processing, ML
2.Pandas & NumPy – Data manipulation
3.Scikit-learn & XGBoost – Machine Learning models
4.Flask – Web application framework
5.Bootstrap – Responsive UI design
6.Joblib – Saving/loading trained ML models
------------------------------------------------------------------------------------------------------------------------------------------

------------------------------------------------------------------------------------------------------------------------------------------
HeartDiseaseApp/
│
├── app  # python flask code  
│    ├── templates/
│    │   ├── index.html
│    │   └── result.html 
│    ├── static/
│    │   ├── css/
│    │   ├── js/
│    ├── main.py         # Main Flask app
├── requirements.txt       # Python packages
├── README.md
├── model/
│   ├── Heart_disease_model.pkl
│   ├── train_columns.pkl
│  
├── notebook/
│   ├── EDA,preprocessing,model trainning # all process in one ipynb 
│   
├── data/
    ├── heart_disease_2020.csv

------------------------------------------------------------------------------------------------------------------------------------------

------------------------------------------------------------------------------------------------------------------------------------------

Installation

1.Clone the repository:

    git clone https://github.com/yourusername/HeartDiseaseApp.git
    cd HeartDiseaseApp


2.Create a virtual environment (optional but recommended):

    python -m venv venv
    source venv/bin/activate   # Linux/Mac
    venv\Scripts\activate      # Windows


3.Install required packages:

    pip install -r requirements.txt

4.Running the App
    
    python app.py


Open browser at http://127.0.0.1:5000/
Fill the form with patient details
Click Predict to see the result

------------------------------------------------------------------------------------------------------------------------------------------

------------------------------------------------------------------------------------------------------------------------------------------

Model Details

1.Algorithms Used:

    -->Logistic Regression(average accuracy ~75%)
    -->Random Forest(best accuracy ~87%) ---> using this model 
    -->XGBoost (accuracy ~85%)

2.Preprocessing:

    -->One-hot encoding for categorical columns (Race, Diabetic)
    -->Ordinal mapping for AgeCategory
    -->Oversampling to balance classes

------------------------------------------------------------------------------------------------------------------------------------------

------------------------------------------------------------------------------------------------------------------------------------------
Future Enhancements

    -->Display probability scores (0–100%) instead of just YES/NO
    -->add charts (SHAP feature importance, heart risk plot)
    -->Deploy on Render/Heroku/Railway for public access
    -->Add user authentication for multiple users
------------------------------------------------------------------------------------------------------------------------------------------

------------------------------------------------------------------------------------------------------------------------------------------
Author

    Name: Saranraj.T
    GitHub: https://github.com/yourusername
    Email: saranrajtsaranrajt27@gmail.come
------------------------------------------------------------------------------------------------------------------------------------------