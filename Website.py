import streamlit as st
import pandas as pd
import joblib
import numpy as np
import plotly.express as px
import plotly.graph_objects as go


model = joblib.load('RFR4.pkl') 
model_columns = joblib.load('model_columns.pkl')  # Columns saved after OHE

# App title
st.set_page_config(page_title="HDB Resale Price Prediction",page_icon="🏠")
st.write("""
# Resale Price Prediction App
This app predicts the **HDB resale price** based on user input!
""")

# Sidebar for user input
st.sidebar.header('Input Features')

# Input form
def user_input_features():
    town = st.sidebar.selectbox('Town',['ANG MO KIO', 'BEDOK', 'BISHAN', 'BUKIT BATOK', 'BUKIT MERAH',
       'BUKIT PANJANG', 'BUKIT TIMAH', 'CENTRAL AREA', 'CHOA CHU KANG',
       'CLEMENTI', 'GEYLANG', 'HOUGANG', 'JURONG EAST', 'JURONG WEST',
       'KALLANG/WHAMPOA', 'MARINE PARADE', 'PASIR RIS', 'PUNGGOL',
       'QUEENSTOWN', 'SEMBAWANG', 'SENGKANG', 'SERANGOON', 'TAMPINES',
       'TOA PAYOH', 'WOODLANDS', 'YISHUN'])
    flat_type = st.sidebar.selectbox('Flat Type',['2 ROOM', '3 ROOM', '4 ROOM', '5 ROOM', 'EXECUTIVE',
       'MULTI-GENERATION'])
    
    flat_model = st.sidebar.selectbox('Flat Model', ['Improved', 'New Generation', 'DBSS', 'Standard', 'Apartment',
       'Simplified', 'Model A', 'Premium Apartment', 'Adjoined flat',
       'Model A-Maisonette', 'Maisonette', 'Type S1', 'Type S2',
       'Model A2', 'Terrace', 'Improved-Maisonette', 'Premium Maisonette',
       'Multi Generation', 'Premium Apartment Loft', '2-room', '3Gen'])
    
    storey_range = st.sidebar.selectbox('Storey Range', ['Low','Mid','High'])
    floor_area_sqm = st.sidebar.slider('Floor Area (sqm)', 30.0, 200.0, 100.0)
    house_age = st.sidebar.slider('House Age (years)', 0, 99, 10)

    data = {
        'town': town,
        'flat_type': flat_type,
        'flat_model':flat_model,
        'storey_range': storey_range,
        'floor_area_sqm': floor_area_sqm,
        'house_age': house_age
    }
    features = pd.DataFrame(data, index=[0])
    return features

df = user_input_features()

st.subheader('User Input Parameters')
st.write(df)
def preprocess_input(input_df):
    categorical_features = ['town', 'flat_type','flat_model', 'storey_range']
    input_df = pd.get_dummies(input_df, columns=categorical_features)
    
    input_df = input_df.reindex(columns=model_columns, fill_value=0)
    st.write("Processed Input DataFrame:", input_df)
    return input_df

processed_input = preprocess_input(df)

prediction = model.predict(processed_input)

st.subheader('Prediction')
st.write(f"The predicted resale price is **${prediction[0]:,.2f}**.")

fig = go.Figure()

fig.add_trace(go.Bar(
    x=["Floor Area (sqm)", "House Age (years)"],
    y=[df['floor_area_sqm'][0], df['house_age'][0]],
    marker_color=['#1f77b4', '#ff7f0e'],
    text=["Floor Area", "House Age"],
    hoverinfo="x+y+text",
))

fig.update_layout(
    title="Impact of Features on Price Prediction",
    xaxis_title="Features",
    yaxis_title="Values",
    showlegend=False
)

st.subheader("Model Feature Importance")
importances = model.feature_importances_
features = model_columns  


importance_df = pd.DataFrame({"Feature": features, "Importance": importances})
importance_df = importance_df.sort_values(by="Importance", ascending=False)

fig2 = px.bar(importance_df, x="Feature", y="Importance", title="Feature Importance")
st.plotly_chart(fig2)


confidence_interval = (prediction[0] - 20000, prediction[0] + 20000)
st.subheader("Prediction Confidence Range")
st.write(f"Confidence Interval: **${confidence_interval[0]:,.2f} - ${confidence_interval[1]:,.2f}**")


