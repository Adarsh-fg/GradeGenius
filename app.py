import streamlit as st  
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pickle as pkl

st.title ('Student Performance Analysis')
st.subheader('Enter the data of the students')

# Create a function to get the data from the user   
# 'Hours_Studied','Attendance','Access_to_Resources_m','Motivation_Level_m'

hours_studied = st.number_input('Enter the number of hours studied', min_value=0, max_value=100, value = 50)
attendance = st.number_input('Enter the attendance percentage', min_value=0, max_value=100, value = 50)
access_to_resources = st.selectbox('Enter the access to resources', ['Low', 'Medium', 'High'])
motivation_level = st.selectbox('Enter the motivation level',['Low', 'Medium', 'High'])

input_data = {'Hours_Studied': hours_studied,
                'Attendance': attendance,
                'Access_to_Resources': access_to_resources,
                'Motivation_Level': motivation_level}

new_data = pd.DataFrame([input_data])

lmh={
    'Low':1,
    'Medium':2,
    'High':3
}

new_data['Access_to_Resources_m'] =new_data['Access_to_Resources'].map(lmh)
new_data['Motivation_Level_m'] =new_data['Motivation_Level'].map(lmh)

df = pd.read_csv('features.csv')
column_list = [col for col in df.columns if col != 'unnamed: 0 ']

new_data = new_data.reindex(columns=column_list, fill_value = 0)

with open('model.pkl', 'rb') as file:
    loaded_model = pkl.load(file)
    
prediction = loaded_model.predict(new_data)

if st.button('Predict'):
    if prediction[0] > 50:
        st.balloons()
    st.write('The predicted score is:', prediction[0])  # Display the predicted score