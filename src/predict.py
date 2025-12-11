import pandas as pd
import joblib

# load a model 
try:
    model = joblib.load("../model/Heart_disease_model.pkl")
    print("✅ model connect sucessfully")
except FileNotFoundError as e:
    print("❌ model connection falild",e)

# column for train dataset
try:
    train_column = joblib.load("../model/train_columns.pkl")
    print("✅ train_model also connect sucessfully")
except FileNotFoundError as e:
    print("❌ train_connection falild",e)

# if user enter a input value extra column or missing column to model optimization of input
def opt_pre(usr_df):
    df = pd.DataFrame([usr_df])

    # handle a missing value
    miss = set(train_column) - set(df.columns)
    for mis_col in miss:
        df[mis_col] = 0
    
    # extra coln value
    extr = set(df.columns) - set(train_column)
    for col_ex in extr:
        df = df.drop(columns=col_ex)

    # reorder the column
    df = df[train_column]
    return df
def predicted(user_input):
    usr_df = opt_pre(user_input)
    model_pre = model.predict(usr_df)[0]
    return "💔 patient  Heart Diease" if model_pre == 1 else "💓 patient no Heart Diease"
if __name__ =="__main__":
    sample_input = {
        'BMI' : 27.7, 
        'Smoking':1, 
        'AlcoholDrinking':0,
        'Stroke':0, 
        'PhysicalHealth':6	,
        'MentalHealth':0,
        'DiffWalking':1,
        'Sex':0, 
        'AgeCategory':77,
        'PhysicalActivity':0,
        'GenHealth':3,
        'SleepTime':12, 
        'Asthma':0, 
        'KidneyDisease':0,
        'SkinCancer':0,
        'Diabetic_No':2,
        'borderline diabetes':0,
        'Diabetic_Yes (during pregnancy)':0,
        'Race_Asian':0,
        'Race_Black':1,
        'Race_Hispanic':0,
        'Race_Other':0,
        'Race_White':0
    }
    print(predicted(sample_input))