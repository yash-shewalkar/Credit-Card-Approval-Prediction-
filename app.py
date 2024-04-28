import streamlit as st
import pandas as pd
from sklearn.preprocessing import StandardScaler
import joblib

# Function to preprocess input data
def preprocess_input_data(input_data):
    # Create a copy of the DataFrame to avoid modifying the original data
    preprocessed_data = input_data.copy()

    # For removing nulls and handling specific values
    preprocessed_data = preprocessed_data.loc[preprocessed_data['Age_Oldest_TL'] != -99999]

    # Here you can add similar preprocessing steps as done for df2

    # Label encoding for categorical features
    mapping = {'SSC': 1, '12TH': 2, 'GRADUATE': 3, 'UNDER GRADUATE': 3, 'POST-GRADUATE': 4, 'OTHERS': 1, 'PROFESSIONAL': 3}
    preprocessed_data['MARITALSTATUS'] = preprocessed_data['MARITALSTATUS'].replace(mapping)

    # One-hot encoding for remaining categorical features
    preprocessed_data = pd.get_dummies(preprocessed_data, columns=['MARITALSTATUS', 'GENDER', 'last_prod_enq2', 'first_prod_enq2'])

    # Standard Scaling for numerical features
    scaler = StandardScaler()
    columns_to_be_scaled = ['Age_Oldest_TL', 'Age_Newest_TL', 'time_since_recent_payment', 'max_recent_level_of_deliq', 'recent_level_of_deliq', 'time_since_recent_enq', 'NETMONTHLYINCOME', 'Time_With_Curr_Empr']
    preprocessed_data[columns_to_be_scaled] = scaler.fit_transform(preprocessed_data[columns_to_be_scaled])

    return preprocessed_data

# Load the model
model = joblib.load("model.pkl")

# Streamlit app
st.title("Credit Risk Prediction App")

# Upload CSV file
uploaded_file = st.file_uploader("Upload CSV file", type=["csv"])

if uploaded_file is not None:
    # Read CSV file
    input_data = pd.read_csv(uploaded_file)

    # Preprocess input data
    preprocessed_data = preprocess_input_data(input_data)

    # Predict target variable
    predicted_values = model.predict(preprocessed_data.drop(columns=['PROSPECTID']))

    # Add predicted values to the DataFrame
    preprocessed_data['Predicted_Target'] = predicted_values

    # Download modified CSV file
    csv_download_link = download_link(preprocessed_data, file_label="Download Predictions", filename="predicted_data.csv", label="Click here to download the predictions CSV file")
    st.markdown(csv_download_link, unsafe_allow_html=True)

# Function to create download link for DataFrame as CSV
def download_link(df, file_label, filename, label):
    csv = df.to_csv(index=False)
    b64 = base64.b64encode(csv.encode()).decode()  # some strings <-> bytes conversions necessary here
    href = f'<a href="data:file/csv;base64,{b64}" download="{filename}">{label}</a>'
    return href
