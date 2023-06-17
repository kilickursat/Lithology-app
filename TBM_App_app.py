import streamlit as st
import pandas as pd
from sklearn.cluster import KMeans
from imblearn.over_sampling import SMOTE
from clover.over_sampling import ClusterOverSampler
from clover.distribution import DensityDistributor
from pycaret.classification import *
import shap
import io
import joblib
import matplotlib.pyplot as plt
import numpy as np
import plotly.express as px


# Set the page title and description
st.title("Soft ground tunnel lithology classification using clustering-guided light gradient boosting machine")
st.write("This app is to identify a soft ground tunnel lithology based on a TBM's operational parameters")
st.write("Created by https://github.com/kilickursat")
#set a subheader

model = load_model('layers-pipeline')


# Data loading options
data_load_option = st.radio("Data Load Option", ("Online", "Batch"))

if data_load_option == "Online":
    # Online data loading
    online_data = st.text_area("Enter data in CSV format")
    if st.button("Load Data"):
        df = pd.read_csv(io.StringIO(online_data))

else:
    # Batch data loading
    uploaded_file = st.file_uploader("classification_model.xlsx", type="xlsx")
    if uploaded_file is not None:
        df = pd.read_excel(uploaded_file)
        st.subheader("Data Information")


        #show the data as a table
        st.dataframe(df)

        #Show statistics on the data
        st.write(df.describe())

        
def run():
    add_selectbox = st.sidebar.selectbox(
    "How would you like to predict?",
    ("Online", "Batch"))

    st.sidebar.info('This app is created to classify a soft ground tunnel lithology')
    #st.sidebar.success('https://www.pycaret.org')
    
    #st.sidebar.image(image_hospital)

    #st.title("Insurance Charges Prediction App")
    
    if add_selectbox == 'Online':
        pressure_gauge1 = st.number_input('Pressure gauge 1 (kPa)', min_value=0, value=0)
        pressure_gauge2 = st.number_input('Pressure gauge 2 (kPa)', min_value=0, value=0)
        pressure_gauge3 = st.number_input('Pressure gauge 3 (kPa)', min_value=0, value=0)
        pressure_gauge4 = st.number_input('Pressure gauge 4 (kPa)', min_value=0, value=0)
        digging_velocity_left = st.number_input('Digging velocity left (mm/min)', min_value=0, value=0)
        digging_velocity_right = st.number_input('Digging velocity right (mm/min)', min_value=0, value=0)
        shield_jack_stroke_left = st.number_input('Shield jack stroke left (mm)', min_value=0, value=0)
        shield_jack_stroke_right = st.number_input('Shield jack stroke right (mm)', min_value=0, value=0)
        propulsion_pressure = st.number_input('Propulsion pressure (MPa)', min_value=0, value=0)
        total_thrust = st.number_input('Total thrust (kN)', min_value=0, value=0)
        cutter_torque = st.number_input('Cutter torque (kNm)', min_value=0, value=0)
        cutterhead_rotation_speed = st.number_input('Cutterhead rotation speed (rpm)', min_value=0, value=0)
        screw_pressure = st.number_input('Screw pressure (MPa)', min_value=0, value=0)
        screw_rotation_speed = st.number_input('Screw rotation speed (rpm)', min_value=0, value=0)
        gate_opening = st.number_input('Gate opening (%)', min_value=0, max_value=100, value=0)
        mud_injection_pressure = st.number_input('Mud injection pressure (MPa)', min_value=0, value=0)
        add_mud_flow = st.number_input('Add mud flow (L/min)', min_value=0, value=0)
        back_in_injection_rate = st.number_input('Back in injection rate (%)', min_value=0, max_value=100, value=0)
        
        output = ""

        output_dict = {'VCS': 0, 'VG': 1,'VSG': 2 }
        output_df = pd.DataFrame([output_dict])
        
if st.button("Train Model"):
            # Prediction on unseen data
            unseen_data = predict_model(model, data=df2)
            st.write("Predicted Data:")
            st.write(unseen_data.head(10))
            
            # Visualization
            st.write("Feature Importance:")
            plot_model(model, plot="feature", display_format="streamlit")
            
            st.write("Confusion Matrix:")
            plot_model(tmodel, plot="confusion_matrix",display_format="streamlit")
            
            st.write("ROC Curve:")
            plot_model(model, plot="auc",display_format="streamlit")
        

           # Explaining the model's predictions using SHAP values
           # https://github.com/slundberg/shap
if st.button("Interpretation"):
            explainer = shap.TreeExplainer(model)
            shap_values = explainer.shap_values(df2)



        #st.write("SHAP Values:")
        #df2 = df2.astype(np.float32)
        #explainer = shap.Explainer(tuned_lightgbm_balanced)
        #shap_values = explainer(df2)
        #shap.summary_plot(shap_values, df2)

        # Save the model as pickle
        #model_filename = "trained_model.pkl"
        #with open(model_filename, "wb") as file:
            #pickle.dump(tuned_lightgbm_balanced, file)
        #st.write(f"Trained model saved as {model_filename}")
