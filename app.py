import streamlit as st
from src.pipeline.service_pipeline import PredictionPipeline, ExampleData



predition_pipeline=PredictionPipeline()

gender=st.selectbox('Gender', ['female', 'male'])
race=st.selectbox('Race or Ethnicity', ['group A', 'group B', 'group C',
                                         'group D', 'group E'])
parental_edu=st.selectbox('Parental Level of Education', ['some high school', 'high school',
                                                           'some college', "associate's degree",
                                                           "bachelor's degree", "master's degree"])
lunch=st.selectbox('Lunch', ['standard', 'free/reduced'])
test_preparation_course=st.selectbox('Test Preparation Course', ['none', 'completed'])




if st.button('Predict'):
    example=ExampleData(gender=gender,
                           race_ethnicity=race,
                           parental_level_education=parental_edu,
                           lunch=lunch,
                           test_preparation_course=test_preparation_course)
    data=example.format_data_as_frame()
    prediction=predition_pipeline.predict(data)
    
    st.write('Predicted Score:', prediction)

    
    