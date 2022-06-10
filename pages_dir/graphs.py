import streamlit as st
import plotly.figure_factory as ff
import plotly.express as px
import plotly.graph_objects as go
from sklearn.metrics import r2_score
from sklearn.metrics import mean_squared_error
import pandas as pd
import numpy as np


@st.cache
def plot_histogram(data, x, height, width):
    """used to display histogram in the data visualization page"""
    # USED
    fig = px.histogram(data, x=x)
    fig.update_layout(bargap=0.2, height=height, width=width)

    return fig


@st.cache(allow_output_mutation=True)
def plot_line(data, x, y, height, width):
    # USED
    fig = px.line(data, x=x, y=y)

    fig.update_layout(bargap=0.05, height=height, width=width)

    return fig


@st.cache
def plot_scatter(data, x, y, height, width, margin,title_text):
    fig = px.scatter(
        data, x=x, y=y,
        # marginal_x='histogram',
        # marginal_y='histogram',
        trendline='ols',
        opacity=.5
    )

    fig.update_layout(bargap=0.05, height=height, width=width, title_text=title_text, margin=dict(t=margin,
                                                                                                  b=margin
                                                                                                  )
                      )
    return fig


@st.cache(hash_funcs={pd.DataFrame: lambda x: x})
def plot_boxplot(data, x, y, height, width, margin, color=None, single_box=False, model_name=None, custom_feature=None,
                 custom_target=None, title_text=None):
    if single_box:
        fig = go.Figure(
            go.Box(
                y=data.loc[(data['name'] == model_name) & (data['custom_features'] == custom_feature) & (
                            data['custom_target'] == custom_target)]['all_scores_cv'].iloc[0],
                name=model_name,
                marker_color='darkblue',
                boxpoints='all',
                jitter=0.3,
                boxmean=True
            )
        )
    else:
        fig = px.box(data, x=x, y=y, color=color)

    fig.update_layout(bargap=0.05, height=height, width=width, title_text=title_text, margin=dict(t=margin,
                                                                                                  b=margin
                                                                                                  )
                      )
    return fig


@st.cache
def plot_countplot(data, x, height, width, margin, title_text=None):
    fig = px.histogram(data, x=x, color=x)
    fig.update_layout(bargap=0.05, height=height, width=width, title_text=title_text, margin=dict(t=margin,
                                                                                                  b=margin
                                                                                                  )
                      )
    return fig


@st.cache
def plot_heatmap(corr_matrix):
    # USED
    fig = ff.create_annotated_heatmap(
        z=np.round(corr_matrix.values, 2),
        x=corr_matrix.index.tolist(),
        y=corr_matrix.columns.values.tolist(),
        hoverinfo="text",
        hovertext=np.round(corr_matrix.values, 2),
        showscale=True,
    )
    fig.update_layout(bargap=0.05, height=550, width=700)

    return fig


@st.cache
def plot_distplot(y_real, y_predict, height, width, margin, title_text=None):
    fig = ff.create_distplot(
        [y_real, y_predict],
        ['Real', 'Predicted'],
        bin_size=150,
        # show_hist=False
    )

    fig.update_layout(bargap=0.05, height=height, width=width, title_text=title_text, margin=dict(t=margin,
                                                                                                  b=margin
                                                                                                  )
                      )

    return fig


@st.cache
def plot_bar(data, x, y, height, width, margin, title_text=None):
    fig = px.bar(data, x=x, y=y, color=x)

    fig.update_layout(bargap=0.05, height=height, width=width, title_text=title_text, margin=dict(t=margin,
                                                                                                  b=margin
                                                                                                  )
                      )

    return fig
