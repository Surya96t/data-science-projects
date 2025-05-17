import pandas as pd
from sklearn.model_selection import train_test_split

from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import MinMaxScaler, OneHotEncoder
from sklearn.impute import SimpleImputer

from sklearn.linear_model import LogisticRegression

from sklearn.metrics import classification_report, accuracy_score

import pickle

def get_clean_data():
    
    
    df = pd.read_csv('./data/heart.csv')

    df['sex'] = df['sex'].map({1: 'male', 0: 'female'})
    df['cp'] = df['cp'].map({1: 'Typical Angina', 2: 'Atypical Angina', 3: 'Non-anginal Pain', 4: 'Asymptomatic'})  
    df['fbs'] = df['fbs'].map({1: 'High', 0: 'Low'})
    df['restecg'] = df['restecg'].map({0: 'Normal', 1: 'ST-T Abnormality', 2: 'Left Ventricular Hypertrophy'})
    df['exang'] = df['exang'].map({1: 'Yes', 0: 'No'})
    df['ca'] = df['ca'].map({0: 'No Vessels Blocked', 1: '1 Vessel Blocked', 2: '2 Vessels Blocked', 3: '3 Vessels Blocked'})
    df['thal'] = df['thal'].map({1: 'Fixed Defect', 2: 'Normal', 3: 'Reversible Defect'})
    df['slope'] = df['slope'].map({0: 'Upsloping', 1: 'Flat', 2: 'Downsloping'})
    
    return df

def data_modeling(data): 
    X = data.drop(['target'], axis=1)
    y = data['target']
    
    # print(X.info())
    # dividing the columns into categorical and numerical columns
    categorical_columns = X.select_dtypes(include=['object']).columns
    numerical_columns = X.select_dtypes(include=['int64', 'float64']).columns  
    
    # splitting the data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42,stratify=y)
    
    numerical_transformer = Pipeline(steps=[
        ('scaler', MinMaxScaler())
    ])
    
    categorical_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='most_frequent')),  
        ('onehot', OneHotEncoder())
    ])  
    
    preprocessor = ColumnTransformer(
        transformers=[
            ('num', numerical_transformer, numerical_columns),
            ('cat', categorical_transformer, categorical_columns)
        ])
    
    pipeline = Pipeline(steps=[('preprocessor', preprocessor), ('LogReg', LogisticRegression())])
    
    
    pipeline.fit(X_train, y_train)
    
    #####################################
    # X_train_transformed = pipeline.named_steps['preprocessor'].transform(X_train)
    #print(X_train_transformed)
    
    # df_transformed = pd.DataFrame(X_train_transformed)
    #print(df_transformed.isnull().sum())
    
    #print(df_transformed.info())
    #########################################
    
    y_pred = pipeline.predict(X_test)
    
    print("Accuracy: ", accuracy_score(y_test, y_pred))
    print("Classification Report: \n ", classification_report(y_test, y_pred))
    
    return pipeline
    
    
def main():
    
    # Load the data
    data = get_clean_data()
    
    # preprocess and model the data
    pipeline = data_modeling(data)
        
    with open('./model/pipeline.pkl', 'wb') as file:
        pickle.dump(pipeline, file)


if __name__ == '__main__':
    main()
    
    
