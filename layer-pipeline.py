from pycaret.classification import load_model, predict_model
import streamlit as st
import pandas as pd
import numpy as np
model = load_model('layers-pipeline')

def predict(model, input_df):
    predictions_df = predict_model(estimator=model, data=input_df)
    predictions = predictions_df['Layers'][0]
    return predictions

def run():

    #from PIL import Image
    #image = Image.open('logo.png')
    #image_hospital = Image.open('hospital.jpg')

    #st.image(image,use_column_width=False)

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

      output_dict = {'VCS': vcs_value, 'VSG': vsg_value, 'VG': vg_value}
      output_df = pd.DataFrame([output_dict])


      if st.button("Predict"):
            output = predict(model=model, input_df=input_df)
            output = '$' + str(output)

      st.success('The output is {}'.format(output))

    if add_selectbox == 'Batch':

        file_upload = st.file_uploader("Upload csv file for predictions", type=["csv"])

        if file_upload is not None:
            data = pd.read_csv(file_upload)
            predictions = predict_model(estimator=model,data=data)
            st.write(predictions)

if __name__ == '__main__':
    run()
