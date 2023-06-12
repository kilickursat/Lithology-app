# -*- coding: utf-8 -*-
"""TBM.ipynb

Automatically generated by Colaboratory.

Original file is located at
    https://colab.research.google.com/drive/1DLaOxMhJA2Uo42CfsEoOXvNEOpejARNh
"""

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

# Set the page title and description
st.title("Soft ground tunnel lithology classification using clustering-guided light gradient boosting machine")
st.write("This app is to identify a soft ground tunnel lithology based on a TBM's operational parameters")
st.write("Created by https://github.com/kilickursat")

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

if "df" in locals():
    # Data preprocessing and sampling
    df2 = df.fillna(0)

    df2 = df2.drop(['RING NU', 'Altitude', 'ExcavationD', 'pitching', 'rolling',
                    'Middle break left and right fold angle (%)', 'Middle break upper and lower folds (%)',
                    ' Geosanth exploration equipment exploration pressure (kN)',
                    'Geoyama Exploration Equipment Exploration Stroke (mm)',
                    'Clay shock injection pressure (MPa)', 'Clay shock flow rateA (L/min)',
                    'Clay shock flow rateB (L/min)', 'Back injection pressure (MPa)', ' Rotation angle (degree)',
                    'Bubble injection pressure (MPa)', 'Back in flow rate of A liquid (L/min)',
                    'Back in flow rate of B liquid (L/min)', 'Excavated Tunnel Length (m)'], axis=1)

    RANDOM_SEED = 142

    @st.cache(allow_output_mutation=True)
    def data_sampling(dataset, frac: float, random_seed: int):
        data_sampled_a = dataset.sample(frac=frac, random_state=random_seed)
        data_sampled_b = dataset.drop(data_sampled_a.index).reset_index(drop=True)
        data_sampled_a.reset_index(drop=True, inplace=True)
        return data_sampled_a, data_sampled_b

    # Add data sampling options
    frac = st.slider("Data Sampling Fraction", 0.0, 1.0, 0.9, 0.1)
    random_seed = st.number_input("Random Seed", value=142)

    # Add data sampling button
    if st.button("Sample Data"):
        df2, data_unseen = data_sampling(df2, frac, random_seed)
        st.write(f"There are {data_unseen.shape[0]} samples for Unseen Data.")

    # Model training and evaluation
    if st.button("Train Model"):
        clovrs = ClusterOverSampler(oversampler=SMOTE(random_state=1),
                                    clusterer=KMeans(random_state=2),
                                    distributor=DensityDistributor(), random_state=3)
        session_2 = setup(df2, target='Layers', session_id=177, log_experiment=False,
                          experiment_name='lithologies2', normalize=True, normalize_method='minmax',
                          transformation=True, transformation_method='quantile', fix_imbalance=True,
                          fix_imbalance_method=clovrs, remove_multicollinearity=True, multicollinearity_threshold=0.6)
        best_model1 = compare_models(sort="F1")
        lightgbm_balanced = create_model('lightgbm', fold=5)
        tuned_lightgbm_balanced = tune_model(lightgbm_balanced, fold=5, optimize="F1")
                # Model interpretation
        interpret_model(tuned_lightgbm_balanced)

        # Prediction on unseen data
        unseen_data = predict_model(tuned_lightgbm_balanced, data=df2)
        st.write("Predicted Data:")
        st.write(unseen_data.head(10))

        # Visualization
        #st.write("Feature Importance:")
        #plot_model(tuned_lightgbm_balanced, plot="feature")

        #st.write("Confusion Matrix:")
        #plot_model(tuned_lightgbm_balanced, plot="confusion_matrix")

        #st.write("ROC Curve:")
        #plot_model(tuned_lightgbm_balanced, plot="auc")
        
        # Visualization
        st.write("Feature Importance:")
        fig_feature = plot_model(tuned_lightgbm_balanced, plot="feature")
        plt.show(fig_feature)
        st.pyplot(fig_feature)

        st.write("Confusion Matrix:")
        fig_confusion_matrix = plot_model(tuned_lightgbm_balanced, plot="confusion_matrix")
        plt.show(fig_confusion_matrix)
        st.pyplot(fig_confusion_matrix)

        st.write("ROC Curve:")
        fig_roc = plot_model(tuned_lightgbm_balanced, plot="auc")
        plt.show(fig_roc)
        st.pyplot(fig_roc)


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
