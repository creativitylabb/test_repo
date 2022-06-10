import io
import missingno as msno

import streamlit as st
import pandas as pd
import plotly.express as px
# import qrcode
# from PIL import Image
#TODO: add libraries psutil, PIL, qrcode
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches


# import functions

def app():
    st.title('Custom Data-Explore')
    dataset = st.file_uploader("Upload a dataset", type=["csv"])

    st.sidebar.header('Import a dataset to view the analysis')
    if dataset:
        # if file_format == 'csv' or use_defo:
        df = pd.read_csv(dataset)

        rows, columns = df.shape
        st.write('Uploaded dataset has ', rows, ' rows and ', columns, ' columns')
        st.write('First 5 rows of the dataset are: ')

        st.dataframe(df.head())
        with st.expander('View entire dataset'):
            st.dataframe(df)

        # img = qrcode.make(df.head())
        # type(img)  # qrcode.image.pil.PilImage
        # img.save("df.png")

        all_menu_items = ['Empty Values', 'Descriptive Analysis',
                       'Distribution of Numerical Columns',
                       'Box Plots']

        all_menu_items = st.sidebar.selectbox("Explore the uploaded dataset", all_menu_items)


        if 'Empty Values' in all_menu_items:
            st.subheader('Missing data information')
            if df.isnull().sum().sum() == 0:
                st.write('Your dataset does not have empty values.')
            else:
                # import missingno as msno
                res = pd.DataFrame(df.isnull().sum()).reset_index()
                res['Percentage'] = round(res[0] / df.shape[0] * 100, 2)
                res['Percentage'] = res['Percentage'].astype(str) + '%'
                res.rename(columns={'index': 'Column Name', 0: 'Count'}, inplace=True)
                st.dataframe(res)

                # fig = plt.figure()

                # ax1 = fig.add_subplot(1, 1, 1)
                # msno.bar(df, orientation="right", ax=ax1)
                # fig.show()
                import numpy as np
                # arr = np.random.normal(1, 1, size=100)
                # fig, ax = plt.subplots()
                #
                # ax.plt(msno.bar(df))
                # ax.hist(arr, bins=20)
                msno.bar(df)

                gray_patch = mpatches.Patch(color='gray', label='Data present')
                white_patch = mpatches.Patch(color='white', label='Data absent ')
                plt.legend(handles=[gray_patch, white_patch])

                # plt.show()
                # fig, axarr = plt.subplots(3)
                # msno.bar(df, ax=axarr[0])
                #
                # # plt.tight_layout()
                st.pyplot(plt)


        if 'Detailed Analysis' in all_menu_items:
            st.subheader('Detailed Analysis:')
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
            if len(numerical_columns) == 0:
                st.write('There is no numerical columns in the data.')
            else:
                selected_num_cols = sidebar_multiselect_container('Choose columns for Box plots:', numerical_columns,
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


    # https://github.com/anarabiyev/EDA_Streamlit_App/blob/master/functions.py


def sidebar_multiselect_container(massage, arr, key):
    container = st.sidebar.container()
    # select_all_button = st.sidebar.checkbox("Select all for " + key + " plots")
    # if select_all_button:
    #     selected_num_cols = container.multiselect(massage, arr, default=list(arr))
    # else:
    selected_num_cols = container.multiselect(massage, arr, default=arr[0])

    return selected_num_cols
