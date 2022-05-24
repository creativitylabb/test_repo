import io

import streamlit as st
import pandas as pd
import plotly.express as px
import qrcode

# import functions


dataset = st.file_uploader("Upload a Dataset", type=["csv"])

st.sidebar.header('Import Dataset to Use Available Features: 👉')


def sidebar_multiselect_container(massage, arr, key):
    container = st.sidebar.container()
    select_all_button = st.sidebar.checkbox("Select all for " + key + " plots")
    if select_all_button:
        selected_num_cols = container.multiselect(massage, arr, default=list(arr))
    else:
        selected_num_cols = container.multiselect(massage, arr, default=arr[0])

    return selected_num_cols
if dataset:
    # if file_format == 'csv' or use_defo:
    df = pd.read_csv(dataset)

    st.subheader('Dataframe:')
    rows, columns = df.shape
    st.write('Uploaded dataset contains ', rows, ' rows and ', columns, ' columns')
    st.write('First 5 rows of dataset are: ')

    st.dataframe(df.head())
    with st.expander('View entire dataset'):
        st.dataframe(df)

    img = qrcode.make(df.head())
    type(img)  # qrcode.image.pil.PilImage
    img.save("df.png")

    all_menu_items = ['Empty Values', 'Descriptive Analysis', 'Target Analysis',
                   'Distribution of Numerical Columns', 'Count Plots of Categorical Columns',
                   'Box Plots', 'Outlier Analysis', 'Variance of Target with Categorical Columns']

    all_menu_items = st.sidebar.selectbox("Explore the uploaded dataset", all_menu_items)


    if 'Empty Values' in all_menu_items:
        st.subheader('NA Value Information:')
        if df.isnull().sum().sum() == 0:
            st.write('Your dataset does not have empty values.')
        else:
            res = pd.DataFrame(df.isnull().sum()).reset_index()
            res['Percentage'] = round(res[0] / df.shape[0] * 100, 2)
            res['Percentage'] = res['Percentage'].astype(str) + '%'
            res.rename(columns={'index': 'Column Name', 0: 'Count'}, inplace=True)
            st.dataframe(res)

    if 'Descriptive Analysis' in all_menu_items:
        st.subheader('Descriptive Analysis:')
        st.dataframe(df.describe())

    if 'Histogram' in all_menu_items:
        st.subheader("Select column to plot histogram for:")
        target_column = st.selectbox("", df.columns, index=len(df.columns) - 1)

        fig = px.histogram(df, x=target_column)
        c1, c2, c3 = st.columns([0.5, 2, 0.5])
        c2.plotly_chart(fig)

    numerical_columns = df.select_dtypes(exclude='object').columns

    if 'Distribution of Numerical Columns' in all_menu_items:

        if len(numerical_columns) == 0:
            st.write('There is no numerical columns in the data.')
        else:
            selected_num_cols = sidebar_multiselect_container('Choose columns for Distribution plots:',
                                                                        numerical_columns, 'Distribution')
            st.subheader('Distribution of numerical columns')
            i = 0
            while (i < len(selected_num_cols)):
                c1, c2 = st.columns(2)
                for j in [c1, c2]:

                    if (i >= len(selected_num_cols)):
                        break

                    fig = px.histogram(df, x=selected_num_cols[i])
                    j.plotly_chart(fig, use_container_width=True)
                    i += 1


    if 'Box Plots' in all_menu_items:
        if len(num_columns) == 0:
            st.write('There is no numerical columns in the data.')
        else:
            selected_num_cols = functions.sidebar_multiselect_container('Choose columns for Box plots:', num_columns,
                                                                        'Box')
            st.subheader('Box plots')
            i = 0
            while (i < len(selected_num_cols)):
                c1, c2 = st.columns(2)
                for j in [c1, c2]:

                    if (i >= len(selected_num_cols)):
                        break

                    fig = px.box(df, y=selected_num_cols[i])
                    j.plotly_chart(fig, use_container_width=True)
                    i += 1



    if 'Variance of Target with Categorical Columns' in all_menu_items:

        df_1 = df.dropna()

        high_cardi_columns = []
        normal_cardi_columns = []

        for i in cat_columns:
            if (df[i].nunique() > df.shape[0] / 10):
                high_cardi_columns.append(i)
            else:
                normal_cardi_columns.append(i)

        if len(normal_cardi_columns) == 0:
            st.write('There is no categorical columns with normal cardinality in the data.')
        else:

            st.subheader('Variance of target variable with categorical columns')
            model_type = st.radio('Select Problem Type:', ('Regression', 'Classification'), key='model_type')
            selected_cat_cols = functions.sidebar_multiselect_container('Choose columns for Category Colored plots:',
                                                                        normal_cardi_columns, 'Category')

            if 'Target Analysis' not in all_menu_items:
                target_column = st.selectbox("Select target column:", df.columns, index=len(df.columns) - 1)

            i = 0
            while (i < len(selected_cat_cols)):

                if model_type == 'Regression':
                    fig = px.box(df_1, y=target_column, color=selected_cat_cols[i])
                else:
                    fig = px.histogram(df_1, color=selected_cat_cols[i], x=target_column)

                st.plotly_chart(fig, use_container_width=True)
                i += 1

            if high_cardi_columns:
                if len(high_cardi_columns) == 1:
                    st.subheader('The following column has high cardinality, that is why its boxplot was not plotted:')
                else:
                    st.subheader(
                        'The following columns have high cardinality, that is why its boxplot was not plotted:')
                for i in high_cardi_columns:
                    st.write(i)

                st.write('<p style="font-size:140%">Do you want to plot anyway?</p>', unsafe_allow_html=True)
                answer = st.selectbox("", ('No', 'Yes'))

                if answer == 'Yes':
                    for i in high_cardi_columns:
                        fig = px.box(df_1, y=target_column, color=i)
                        st.plotly_chart(fig, use_container_width=True)
# https://github.com/anarabiyev/EDA_Streamlit_App/blob/master/functions.py
