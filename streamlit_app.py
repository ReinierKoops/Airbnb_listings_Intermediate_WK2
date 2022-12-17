import gdown
import pandas as pd
import numpy as np
import streamlit as st
import sklearn
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor


@st.cache
def get_data():
    # Download file from Google Drive
    # This file is based on data from: http://insideairbnb.com/get-the-data/
    file_id_1 = "1rsxDntx9CRSyDMy_fLHEI5Np4lB153sa"
    downloaded_file_1 = "listings.pkl"
    gdown.download(id=file_id_1, output=downloaded_file_1)

    # Read a Python Pickle file
    return pd.read_pickle("listings.pkl")


def create_model(X, y):
    # Set seed for reproducibility
    SEED = 42

    # Split up dataset into train and test, of which we split up the test.
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.8, random_state=SEED
    )

    # Create a classifier algorithm + its predictions
    model = RandomForestRegressor()  # Our algorithm

    model.fit(  # Train it ("Learn the material")
        X_train[["amenities", "accommodates", "instant_bookable"]],
        np.squeeze(y_train),
    )

    return model


df = get_data()
X, y = (
    df_list[["amenities", "accommodates", "instant_bookable"]],
    df_list[["host_reported_average_tip"]],
)
model = create_model(X, y)

st.title("Week 2: The Airbnb dataset of Amsterdam")
st.markdown(
    "The dataset contains modifications with regards to the original for illustrative & learning purposes"
)

amenities = st.slider('How many amenities does the listing have?', 0, X["amenities"].max(), 20)
accommodates = st.slider('How many people does the listing accommodate?', 1, X["accommodates"].max(), 2)
instant_bookable = st.radio(
    "Is the listing instantly bookable?",
    ("True", "False"))
instant_bookable = 1 if instant_bookable == "True" else 0

user_input = [[amenities, accommodates, instant_bookable]]

if st.button('Predict?'):
    st.write("The model predicts that the average tip for this listing is:", model.predict(user_input).round(2))
