import streamlit as st
import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
import joblib

# Load the pre-trained KNN model from the pickle file
vec, loaded_model = joblib.load('knn_model1.joblib')

@st.cache_data
def vectorize_dataset(train_df, test_df):

    train_df['symptoms'] = train_df.iloc[:, 1:].astype(str).apply(lambda x: ', '.join(x), axis=1)
    test_df['symptoms'] = test_df.iloc[:, 1:].astype(str).apply(lambda x: ', '.join(x), axis=1)
    vec = CountVectorizer()
    matrix = vec.fit_transform(train_df['symptoms'])
    train_df = pd.DataFrame(matrix.toarray(), columns=vec.get_feature_names_out(), index=train_df['Disease'])
    train_df['Disease'] = train_df.index

    matrix = vec.transform(test_df['symptoms'])
    test_df = pd.DataFrame(matrix.toarray(), columns=vec.get_feature_names_out(), index=test_df['Disease'])
    test_df['Disease'] = test_df.index

    return train_df, test_df

# Assuming you have train_df and test_df
# Replace 'train_df' and 'test_df' with your actual DataFrames
train_df = pd.read_csv('C:/Users/Archana/OneDrive/Desktop/CAPSTONE PORJECT/DATASETS/Disease prediction - symptom based/training_data.csv')
test_df = pd.read_csv('C:/Users/Archana/OneDrive/Desktop/CAPSTONE PORJECT/DATASETS/Disease prediction - symptom based/testing_data.csv')
train_df, test_df = vectorize_dataset(train_df, test_df)

def predict_disease(selected_symptoms):
    # selected_indices = [train_df.columns.tolist().index(symptom) for symptom in selected_symptoms]
    # input_data = np.zeros(len(train_df.columns) - 1)
    # input_data[selected_indices] = 1
    prediction = loaded_model.predict(selected_symptoms) #input_data
    return prediction[0]

st.title("Disease Prediction")
st.markdown('**Objective**: Given symptoms, predict the disease.')

# Display checkboxes for symptoms
symptoms = train_df.columns.tolist()[:-1]  # Exclude 'Disease' column
selected_symptoms = st.multiselect('Select Symptoms:', symptoms)
print(selected_symptoms)
if st.button('Predict'):
    if len(selected_symptoms) == 0:
        st.error('Please select at least one symptom.')
    else:
        # Ensure that selected_symptoms match the columns in the training data
        selected_symptoms = [symptom for symptom in selected_symptoms if symptom in train_df.columns.tolist()] #
        if len(selected_symptoms) == 0:
            st.error('Please select valid symptoms.')
        else:
            predicted_disease = predict_disease(vec.transform([','.join(selected_symptoms)]).toarray())
            st.success(f'Predicted Disease: {predicted_disease}')
