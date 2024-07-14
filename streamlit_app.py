import streamlit as st
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

# Set up the Streamlit app
st.title("Exploratory Data Analysis (EDA) App")
st.write("Upload a CSV file to perform EDA")

# File uploader
uploaded_file = st.file_uploader("Choose a CSV file", type="csv")

if uploaded_file is not None:
    # Read the CSV file
    df = pd.read_csv(uploaded_file)
    
    # Display the dataframe
    st.subheader("DataFrame")
    st.write(df)

    # Display basic information
    st.subheader("Basic Information")
    st.write("Number of rows: ", df.shape[0])
    st.write("Number of columns: ", df.shape[1])
    
    # Display column names
    st.write("Column Names:")
    st.write(df.columns.tolist())

    # Display data types
    st.subheader("Data Types")
    st.write(df.dtypes)
    
    # Display summary statistics
    st.subheader("Summary Statistics")
    st.write(df.describe())

    # Display missing values
    st.subheader("Missing Values")
    st.write(df.isnull().sum())

    # # Display correlation matrix
    # st.subheader("Correlation Matrix")
    # corr = df.corr()
    # sns.heatmap(corr, annot=True, cmap="coolwarm")
    # st.pyplot(plt)

    # Select columns for scatter plot
    st.subheader("Scatter Plot")
    cols = df.columns.tolist()
    x_col = st.selectbox("Select X-axis column", cols)
    y_col = st.selectbox("Select Y-axis column", cols)
    if x_col and y_col:
        sns.scatterplot(data=df, x=x_col, y=y_col)
        st.pyplot(plt)
    
    # Select column for histogram
    st.subheader("Histogram")
    hist_col = st.selectbox("Select column for histogram", cols)
    if hist_col:
        sns.histplot(df[hist_col], kde=True)
        st.pyplot(plt)
    

    st.subheader("Box Plot")
    box_col = st.selectbox("Select column for box plot", cols)
    if box_col:
        fig, ax = plt.subplots()
        sns.boxplot(y=df[box_col], ax=ax)
        ax.set_title(f'Box Plot of {box_col}')
        st.pyplot(fig)
    
    # Select columns for pair plot
    st.subheader("Pair Plot")
    pair_cols = st.multiselect("Select columns for pair plot", cols)
    if len(pair_cols) > 1:
        sns.pairplot(df[pair_cols])
        st.pyplot(plt)

