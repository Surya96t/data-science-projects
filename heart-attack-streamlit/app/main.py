import streamlit as st
import pickle
import pandas as pd 
import numpy as np
import plotly.graph_objects as go

def get_clean_data():
    df = pd.read_csv('./data/heart-disease.csv')

    df['sex'] = df['sex'].map({1: 'male', 0: 'female'})
    df['cp'] = df['cp'].map({1: 'Typical Angina', 2: 'Atypical Angina', 3: 'Non-anginal Pain', 4: 'Asymptomatic'})  
    df['fbs'] = df['fbs'].map({1: 'High', 0: 'Low'})
    df['restecg'] = df['restecg'].map({0: 'Normal', 1: 'ST-T Abnormality', 2: 'Left Ventricular Hypertrophy'})
    df['exang'] = df['exang'].map({1: 'Yes', 0: 'No'})
    df['ca'] = df['ca'].map({0: 'No Vessels Blocked', 1: '1 Vessel Blocked', 2: '2 Vessels Blocked', 3: '3 Vessels Blocked'})
    df['thal'] = df['thal'].map({1: 'Fixed Defect', 2: 'Normal', 3: 'Reversible Defect'})
    df['slope'] = df['slope'].map({0: 'Upsloping', 1: 'Flat', 2: 'Downsloping'})
    
    df.dropna(inplace=True)
    
    return df

def add_sidebar():
    st.sidebar.title("User Input Features")
    
    data = get_clean_data()
    data = data.drop(['target'], axis=1)  # drop the target column
    categorical_columns = data.select_dtypes(include=['object']).columns
    numerical_columns = data.select_dtypes(include=['int64', 'float64']).columns  
    
    
    # List of column names with descriptive labels
    column_labels = [
        ("Age", "age"),
        ("Sex", "sex"),
        ("Chest Pain Type, (cp)", "cp"),
        ("Resting Blood Pressure (mm Hg), (trestbps)", "trestbps"),
        ("Serum Cholesterol (mg/dl), (chol)", "chol"),
        ("Fasting Blood Sugar > 120 mg/dl, (fbs)", "fbs"),
        ("Resting Electrocardiographic Results, (restecg)", "restecg"),
        ("Maximum Heart Rate Achieved, (thalach)", "thalach"),
        ("Exercise-Induced Angina, (exang)", "exang"),
        ("Depression Induced by Exercise Relative to Rest, (oldpeak)", "oldpeak"),
        ("Slope of the Peak Exercise ST Segment, (slope)", "slope"),
        ("Number of Major Vessels Colored by Fluoroscopy, (ca)", "ca"),
        ("Thalassemia, (thal)", "thal"),
    ]
    
    # create dictionary of column labels for chart later on 
    input_dict = {}
    
    # loop through the column labels and add a selectbox for each feature
    for label, column in column_labels:
        if column in categorical_columns:
            input_dict[column] = st.sidebar.selectbox(label, data[column].unique())
        else:
            input_dict[column] = st.sidebar.slider(
                label, 
                min_value=float(data[column].min()), 
                max_value=float(data[column].max()), 
                value=float(data[column].mean())
            )

    return input_dict

def get_scaled_values(input_dict):
    data = get_clean_data()
    
    X = data[['age', 'trestbps', 'chol', 'thalach', 'oldpeak']]
    
    scaled_dict = {}
    
    for key, value in input_dict.items():
        if key not in X.columns:
            scaled_dict[key] = None
            continue
        max_val = X[key].max()
        min_val = X[key].min()
        scaled_value = (value - min_val) / (max_val - min_val)
        scaled_dict[key] = scaled_value
        
    return scaled_dict

def get_radar_chart(input_data):
    
    input_data = get_scaled_values(input_data)
    
    categories = ['Age', 'Resting Blood Pressure (mm Hg)',
                  'Serum Cholesterol (mg/dl)', 'Maximum Heart Rate Achieved',
                  'Depression Induced by Exercise Relative to Rest']

    fig = go.Figure()

    fig.add_trace(go.Scatterpolar(
        r = [
            input_data['age'],
            input_data['trestbps'],
            input_data['chol'],
            input_data['thalach'],
            input_data['oldpeak']
        ],
        theta=categories,
        fill='toself',
        name='User Input'
    ))

    fig.update_layout(
    polar=dict(
        radialaxis=dict(
        visible=True,
        range=[0, 1]
        )),
    showlegend=True
    )

    return fig
    
def get_pipeline():
    pipeline = pickle.load(open('./model/pipeline.pkl', 'rb'))
    return pipeline

def add_predictions(input_data, input_pipeline):
    
    pipeline = input_pipeline
   
    input_df = pd.DataFrame([input_data])
    
    prediction = pipeline.predict(input_df)
    prediction_proba = pipeline.predict_proba(input_df) 
    # print(prediction)
    # st.write(input_df)
    # st.write(preprocessed_df)
    # st.write(prediction)
    # st.write(prediction_proba)
    
    st.subheader("Prediction")
    st.write("The model predicts the following:")
    
    if prediction == 1:
        st.write("<span class='diagnosis heart_disease'> Heart Disease </span>", unsafe_allow_html=True)
    else:
        st.write("<span class='diagnosis no_heart_disease'> No Heart Disease </span>", unsafe_allow_html=True)
        
    st.write("Probability of not Heart Disease: ", prediction_proba[0][0].round(2))
    st.write("Probability of having Heart Disease: ", prediction_proba[0][1].round(2))
    
    st.write("This app is for educational purposes only. Please consult a doctor for medical advice.")
    

def main():
    # setting the page configurations
    st.set_page_config(
        page_title="Heart Disease Predictor",
        page_icon=":heart:",
        layout="wide",
        initial_sidebar_state="expanded" 
    )
    
    with open("./assets/style.css") as f:
        st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)

    # st.title("Heart Disease Predictor") 
    with st.container():
        st.title("Heart Disease Predictor")
        st.write("This is a simple app to predict heart disease using a logistic regression model.")

    input_data = add_sidebar()
    
    
#### Columns ####
    pipeline = get_pipeline()
    col1, col2 = st.columns([3,1]) # column widths column ratio 4:1
    
    with col1:
        radar_chart = get_radar_chart(input_data)
        st.plotly_chart(radar_chart)
        
    with col2:
        add_predictions(input_data, pipeline)
        
    
        
    st.subheader("Feature Importance")
    # get numerical features names
    num_features = ['age', 'trestbps', 'chol', 'thalach', 'oldpeak']
    
    # get one hot encoded features names
    ohe_feature_names = pipeline.named_steps['preprocessor'].named_transformers_['cat'].named_steps['onehot'].get_feature_names_out()
    
    # combining the features 
    feature_names = num_features + ohe_feature_names.tolist()
    # st.write(feature_names)
    
    
    model = pipeline.named_steps['LogReg']
    coefficients = model.coef_[0]   
    
    feature_importance_df = pd.DataFrame({
        'Feature': feature_names,
        'Coefficient': coefficients
    })
    
    feature_importance_df['Absolute Coefficient'] = np.abs(feature_importance_df['Coefficient'])
    feature_importance_df = feature_importance_df.sort_values(by='Absolute Coefficient', ascending=False).reset_index(drop=True)
    st.write(feature_importance_df)

if __name__ == '__main__':
    main()