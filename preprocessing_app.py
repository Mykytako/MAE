import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import LabelEncoder, StandardScaler, MinMaxScaler
import pickle

# Define a function to handle missing values
def handle_missing_values(df, column, method, custom_value=None):
    try:
        if method == "Drop":
            df.dropna(subset=[column], inplace=True)
        elif method == "Fill with Mean":
            df[column].fillna(df[column].mean(), inplace=True)
        elif method == "Fill with Mode":
            df[column].fillna(df[column].mode()[0], inplace=True)
        elif method == "Custom Input":
            if custom_value is not None:
                df[column].fillna(custom_value, inplace=True)
            else:
                st.write("Please provide a custom value")
        return True
    except:
        st.write("Please choose a different option. Missing values have not been handled")
        return False

# Define a function to cap outliers
def cap_outliers(df, column, factor):
    q1, q3 = df[column].quantile([0.25, 0.75])
    iqr = q3 - q1
    top_boundary = q3 + factor * iqr
    bottom_boundary = q1 - factor * iqr
    df[column] = np.where(df[column] > top_boundary, top_boundary, df[column])
    df[column] = np.where(df[column] < bottom_boundary, bottom_boundary, df[column])

def save_progress():
    with open('progress.pkl', 'wb') as f:
        pickle.dump({
            'data': st.session_state.data,
            'mapping': st.session_state.mapping,
            'label_encoded_cols': st.session_state.label_encoded_cols,
            'onehot_encoded_cols': st.session_state.onehot_encoded_cols,
            'handled_missing': st.session_state.handled_missing,
            'capped_outliers': st.session_state.capped_outliers
        }, f)

def load_progress():
    try:
        with open('progress.pkl', 'rb') as f:
            progress = pickle.load(f)
            st.session_state.data = progress['data']
            st.session_state.mapping = progress['mapping']
            st.session_state.label_encoded_cols = progress['label_encoded_cols']
            st.session_state.onehot_encoded_cols = progress['onehot_encoded_cols']
            st.session_state.handled_missing = progress['handled_missing']
            st.session_state.capped_outliers = progress['capped_outliers']
    except FileNotFoundError:
        pass

# Initialize session state
if 'data' not in st.session_state:
    st.session_state.data = None

if 'mapping' not in st.session_state:
    st.session_state.mapping = {}

if 'label_encoded_cols' not in st.session_state:
    st.session_state.label_encoded_cols = []

if 'onehot_encoded_cols' not in st.session_state:
    st.session_state.onehot_encoded_cols = []

if 'handled_missing' not in st.session_state:
    st.session_state.handled_missing = []

if 'capped_outliers' not in st.session_state:
    st.session_state.capped_outliers = []

# Load progress from file
load_progress()

def main():
    st.title("Preprocessing Dashboard")

    # Upload dataset
    file = st.file_uploader("Upload your dataset", type=["csv", "xlsx"])
    if file is not None:
        if file.name.endswith('.csv'):
            st.session_state.data = pd.read_csv(file)
        else:
            st.session_state.data = pd.read_excel(file)
        save_progress()

    if st.session_state.data is not None:
        df = st.session_state.data.copy()

        # Apply existing mappings
        for col, mapping in st.session_state.mapping.items():
            df[col] = df[col].map(mapping)

        # Apply existing label encodings
        if st.session_state.label_encoded_cols:
            le = LabelEncoder()
            for col in st.session_state.label_encoded_cols:
                df[col] = le.fit_transform(df[col])

        # Apply existing one hot encodings
        if st.session_state.onehot_encoded_cols:
            df = pd.get_dummies(df, columns=st.session_state.onehot_encoded_cols)

        # Apply existing missing value handling
        for (col, method, custom_value) in st.session_state.handled_missing:
            handle_missing_values(df, col, method, custom_value)

        # Apply existing outlier capping
        for (col, factor) in st.session_state.capped_outliers:
            cap_outliers(df, col, factor)

        # Display EDA
        st.header("Exploratory Data Analysis")

        # Shape of dataset
        st.subheader("Shape of the dataset")
        st.write(df.shape)

        # Summary of dataset
        st.subheader("Summary of the dataset")
        st.write(df.describe())

        # Data types
        st.subheader("Data Types")
        st.write(df.dtypes)

        # Missing values
        st.subheader("Missing Values")
        st.write(df.isnull().sum())

        # Number of duplicates
        st.subheader("Number of Duplicates")
        st.write(df.duplicated().sum())

        # Detect outliers
        st.subheader("Outlier Detection")
        numeric_cols = df.select_dtypes(include=['float64', 'int64']).columns
        outlier_columns = []
        for col in numeric_cols:
            q1, q3 = df[col].quantile([0.25, 0.75])
            iqr = q3 - q1
            top_boundary = q3 + 1.5 * iqr
            bottom_boundary = q1 - 1.5 * iqr
            if ((df[col] < bottom_boundary) | (df[col] > top_boundary)).any():
                outlier_columns.append(col)

        st.write("Columns with outliers:", outlier_columns)
        selected_outlier_col = st.selectbox("Select a column to view outliers", outlier_columns)
        if selected_outlier_col:
            fig, ax = plt.subplots()
            sns.boxplot(x=df[selected_outlier_col], ax=ax)
            st.pyplot(fig)

        # First few rows
        st.subheader("First Few Rows")
        st.write(df.head())

        # Value counts
        st.subheader("Value Counts of Selected Column")
        selected_val_count_col = st.selectbox("Select a column for value counts", df.columns)
        if selected_val_count_col:
            st.write(df[selected_val_count_col].value_counts())
            fig, ax = plt.subplots()
            df[selected_val_count_col].value_counts().plot(kind='bar', ax=ax)
            st.pyplot(fig)

        # Preprocessing steps
        st.header("Preprocessing Steps")

        # Drop duplicates
        st.subheader("Drop Duplicates")
        if st.button("Drop Duplicates"):
            df.drop_duplicates(inplace=True)
            st.session_state.data = df.copy()
            save_progress()
            st.write("Duplicates dropped. Current shape:", df.shape)

        # Fill or drop missing values
        st.subheader("Handle Missing Values")
        missing_cols = df.columns[df.isnull().any()].tolist()
        selected_missing_col = st.selectbox("Select column with missing values", missing_cols)
        if selected_missing_col:
            missing_method = st.selectbox("Select method to handle missing values", ["Drop", "Fill with Mean", "Fill with Mode", "Custom Input"])
            custom_value = None
            if missing_method == "Custom Input":
                custom_value = st.text_input("Enter custom value")
            if st.button("Handle Missing Values"):
                if handle_missing_values(df, selected_missing_col, missing_method, custom_value):
                    st.session_state.data = df.copy()
                    st.session_state.handled_missing.append((selected_missing_col, missing_method, custom_value))
                    save_progress()
                    st.write("Missing values handled.")

        # Encode selected columns
        st.subheader("Encode Columns")
        label_encode_cols = st.multiselect("Select columns to encode with Label Encoder", df.columns)
        onehot_encode_cols = st.multiselect("Select columns to encode with One Hot Encoder", df.columns)
        if st.button("Encode Columns"):
            if label_encode_cols:
                le = LabelEncoder()
                for col in label_encode_cols:
                    df[col] = le.fit_transform(df[col])
                    if col not in st.session_state.label_encoded_cols:
                        st.session_state.label_encoded_cols.append(col)
            if onehot_encode_cols:
                df = pd.get_dummies(df, columns=onehot_encode_cols)
                for col in onehot_encode_cols:
                    if col not in st.session_state.onehot_encoded_cols:
                        st.session_state.onehot_encoded_cols.append(col)
            st.session_state.data = df.copy()
            save_progress()
            st.write("Columns encoded.")

        # Cap outliers
        st.subheader("Cap Outliers")
        selected_outlier_cols = st.multiselect("Select columns to cap outliers", outlier_columns)
        if selected_outlier_cols:
            outlier_factor = st.slider("Select outlier capping factor", 1.0, 3.0, 1.5)
            if st.button("Cap Outliers"):
                for col in selected_outlier_cols:
                    cap_outliers(df, col, outlier_factor)
                    st.session_state.capped_outliers.append((col, outlier_factor))
                st.session_state.data = df.copy()
                save_progress()
                st.write("Outliers capped.")

        # Map custom values
        st.subheader("Map Custom Values")
        map_col = st.selectbox("Select column to map values", df.columns)
        if map_col:
            map_dict = {}
            unique_values = df[map_col].unique().tolist()
            for val in unique_values:
                new_val = st.text_input(f"Enter new value for {val}", key=f"map_{val}")
                map_dict[val] = new_val
            if st.button("Map Values"):
                st.session_state.mapping[map_col] = map_dict
                df[map_col] = df[map_col].map(map_dict)
                st.session_state.data = df.copy()
                save_progress()
                st.write("Values mapped.")

        # Scale selected columns
        st.subheader("Scale Columns")
        scale_cols = st.multiselect("Select columns to scale", df.select_dtypes(include=['float64', 'int64']).columns.tolist())
        if scale_cols:
            scaler_method = st.selectbox("Select scaler method", ["Standard Scaler", "Min Max Scaler"])
            if st.button("Scale Columns"):
                if scaler_method == "Standard Scaler":
                    scaler = StandardScaler()
                elif scaler_method == "Min Max Scaler":
                    scaler = MinMaxScaler()
                df[scale_cols] = scaler.fit_transform(df[scale_cols])
                st.session_state.data = df.copy()
                save_progress()
                st.write("Columns scaled.")

        # Display preprocessed EDA
        st.header("Preprocessed Exploratory Data Analysis")

        st.subheader("Shape of the dataset")
        st.write(df.shape)

        st.subheader("Summary of the dataset")
        st.write(df.describe())

        st.subheader("Data Types")
        st.write(df.dtypes)

        st.subheader("Missing Values")
        st.write(df.isnull().sum())

        st.subheader("Number of Duplicates")
        st.write(df.duplicated().sum())

        st.subheader("Outlier Detection")
        outlier_columns = []
        for col in numeric_cols:
            q1, q3 = df[col].quantile([0.25, 0.75])
            iqr = q3 - q1
            top_boundary = q3 + 1.5 * iqr
            bottom_boundary = q1 - 1.5 * iqr
            if ((df[col] < bottom_boundary) | (df[col] > top_boundary)).any():
                outlier_columns.append(col)
        st.write("Columns with outliers:", outlier_columns)

        st.subheader("First Few Rows")
        st.write(df.head())

        st.subheader("Value Counts of Selected Column")
        selected_val_count_col = st.selectbox("Select a column for value counts (after preprocessing)", df.columns)
        if selected_val_count_col:
            st.write(df[selected_val_count_col].value_counts())
            fig, ax = plt.subplots()
            df[selected_val_count_col].value_counts().plot(kind='bar', ax=ax)
            st.pyplot(fig)

if __name__ == "__main__":
    main()
