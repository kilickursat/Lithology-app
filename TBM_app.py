import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from imblearn.over_sampling import SMOTE
from clover.over_sampling import ClusterOverSampler
from clover.distribution import DensityDistributor
from pycaret.classification import *
import shap
import io
import pickle

# Set the page title and description
st.title("Machine Learning Web App")
st.write("This is a web application for your machine learning model.")

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

    # Feature selection
    selected_features = st.multiselect("Select Features", df2.columns.tolist())

    df2 = df2[selected_features]

    df2 = df2.dropna()

    # Add data sampling options
    frac = st.slider("Data Sampling Fraction", 0.0, 1.0, 0.9, 0.1)
    random_seed = st.number_input("Random Seed", value=142)

    # Add data sampling button
    if st.button("Sample Data"):
        df2 = df2.sample(frac=frac, random_state=random_seed)
        st.write(f"Sampled Data Shape: {df2.shape}")

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
        evaluate_model(lightgbm_balanced)

        # Model interpretation
        interpret_model(tuned_lightgbm_balanced)

        # Prediction on unseen data
        unseen_data = predict_model(tuned_lightgbm_balanced, data=df2)
        st.write("Predicted Data:")
        st.write(unseen_data.head(10))

        # Visualization
        st.write("Feature Importance:")
        plot_model(tuned_lightgbm_balanced, plot="feature")

        st.write("Confusion Matrix:")
        plot_model(tuned_lightgbm_balanced, plot="confusion_matrix")

        st.write("ROC Curve:")
        plot_model(tuned_lightgbm_balanced, plot="auc")

        st.write("SHAP Values:")
        explainer = shap.Explainer(tuned_lightgbm_balanced)
        shap_values = explainer(df2)
        shap.summary_plot(shap_values, df2)

        # Save the model as pickle
        model_filename = "trained_model.pkl"
        with open(model_filename, "wb") as file:
            pickle.dump(tuned_lightgbm_balanced, file)
        st.write(f"Trained model saved as {model_filename}")

