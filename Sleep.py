# This is a sample Python script.
#import pickle
import joblib as jl
import pandas as pd
import streamlit as st

st.header("An artificial intelligence tool for predicting sleep disturbance for university students after analyzing lifestyle, sports, and psychological health: an externally validated study")
st.sidebar.title("Selection of Parameters")
st.sidebar.markdown("Picking up parameters")

age = st.sidebar.slider("Age (years)", 16, 35)
gender = st.sidebar.selectbox("Gender", ("Male", "Female"))
grade = st.sidebar.selectbox("Grade", ("Grade one", "Grade two", "Grade three", "Grade four"))
drinking = st.sidebar.selectbox("Drinking", ("Not", "Abstained from drinking", "Current drinker"))
barbecue = st.sidebar.selectbox("Loving barbecue", ("No", "Yes"))
vegetable = st.sidebar.selectbox("Loving vegetable", ("No", "Yes"))
sedentary_time = st.sidebar.selectbox("Sedentary time per day (h)", ("<1", "1~3", "3~6", ">6"))
Chronic_disease = st.sidebar.selectbox("Chronic disease", ("No", "Yes"))
GAD_7fj = st.sidebar.selectbox("Severity of anxiety", ("None", "Mild", "Moderate", "Severe"))
PHQ_9fj = st.sidebar.selectbox("Severity of depression", ("None", "Mild", "Moderate", "Moderate to severe", "Severe"))
DASSstressjt = st.sidebar.slider("Stress score", 0, 40)

if st.button("Submit"):
    Xgbc_clf = jl.load("Xgbc_clf_final_round.pkl")
    x = pd.DataFrame([[gender, grade, drinking, barbecue, vegetable, sedentary_time, Chronic_disease, GAD_7fj, PHQ_9fj, age, DASSstressjt]],
                     columns=['gender', 'grade', 'drinking', 'barbecue', 'vegetable', 'sedentary_time', 'Chronic_disease', 'GAD_7fj', 'PHQ_9fj', 'age', 'DASSstressjt'])

    x = x.replace(["Male", "Female"], [1, 2])
    x = x.replace(["Grade one", "Grade two", "Grade three", "Grade four"], [1, 2, 3, 4])
    x = x.replace(["Not", "Abstained from drinking", "Current drinker"], [1, 2, 3])
    x = x.replace(["No", "Yes"], [0, 1])
    # x = x.replace(["是的", "不是"], [1, 0])
    x = x.replace(["<1", "1~3", "3~6", ">6"], [1, 2, 3, 4])
    # x = x.replace(["否", "是"], [0, 1])
    x = x.replace(["None", "Mild anxiety", "Moderate anxiety", "Severe anxiety"], [1, 2, 3, 4])
    x = x.replace(["None", "Mild", "Moderate", "Moderate to severe", "Severe"], [1, 2, 3, 4, 5])

    # Get prediction
    prediction = Xgbc_clf.predict_proba(x)[0, 1]
        # Output prediction
    st.success(f"Probability of developing PerCI: {'{:.2%}'.format(round(prediction, 5))}")
    if prediction < 0.437:
        st.success(f"Risk group: low-risk group")
    else:
        st.success(f"Risk group: High-risk group")
    if prediction < 0.437:
        st.markdown(f"The low-risk population requires proactive measures to maintain and enhance their already good sleep quality. Encouraging a healthy sleep routine and regular exercise can further optimize their sleep patterns. Promoting stress management techniques such as mindfulness meditation or relaxation exercises can be beneficial. It is also important to offer strategies for maintaining a balanced lifestyle, including setting boundaries between study and leisure time, and fostering healthy coping mechanisms for stress.")
    else:
        st.markdown(f"For the high-risk population with poor sleep quality, targeted intervention strategies should be implemented. Firstly, providing education on sleep hygiene is crucial. This includes promoting consistent sleep schedules, creating a conducive sleep environment, avoiding stimulants close to bedtime, and incorporating relaxation techniques before sleep. Additionally, cognitive-behavioral therapy for insomnia can be employed to address psychological factors affecting sleep. This may involve identifying and modifying negative sleep thoughts, implementing behavioral techniques like stimulus control and sleep restriction, and enhancing relaxation skills.")
st.subheader('Model information')
st.markdown('The eXGBM technique was employed to craft the artificial intelligence model, fortified by an in-depth analysis of an extensive pool of 1882 university students. Moreover, the model underwent rigorous external validation on a cohort of 361 university students, which greatly enhanced its credibility. The model showcased an area under the curve of 0.779 (95%CI: 0.728-0.830), dignifying its robust predictability. However, it is pivotal to acknowledge that while the artificial intelligence model furnishes risk assessments and recommendations, an optimal therapeutic approach ought to factor in the profound expertise of healthcare professionals and the unique contextual intricacies of individual patients.')
