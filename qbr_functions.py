import pandas as pd
import numpy as np
import streamlit as st
import plotly.graph_objects as go


# define some functions
@st.experimental_memo
def data_loader(first_data):
    data = pd.read_csv(first_data)
    data['date'] = pd.to_datetime(data['CONV_CREATED'])
    data['usage'] = (data['CLICK'] / data['VIEW']) * 100
    data['usage'].fillna(0, inplace=True)
    return data


@st.cache
def make_str(text):
    txt = str(text)
    return txt


@st.cache
def date_splitter(data_set, splitter):
    splitter1 = pd.to_datetime(splitter)
    pre = data_set[data_set['date'] < splitter1].copy()
    post = data_set[data_set['date'] >= splitter1].copy()
    dset = [pre, post]
    return dset


@st.cache
def usage_spliter(data_set, splitter):
    high = data_set[data_set['usage'] >= splitter]
    low = data_set[data_set['usage'] < splitter]
    dset = [high, low]
    return dset


def red_or_green(val):
    color = 'red' if val > 0 else 'green'
    return f'background-color: {color}'


def random_dates(start, end, n=10):
    start_u = start.value // 10 ** 9
    end_u = end.value // 10 ** 9

    return pd.to_datetime(np.random.randint(start_u, end_u, n), unit='s')


def make_metric(name, val='N/A', change='N/A', d_color='normal'):
    st.metric(name, val, delta=change, delta_color=d_color)


@st.cache
def diverging_sentiment(data_file, company_or_agent='company', intents_or_agents='intents', order_by=-2, top=10):
    # add the " " to make the print-out work
    company_or_agent = '"' + company_or_agent + '"'
    # convert input to column name
    if intents_or_agents == 'Intents':
        intents_or_agents = 'MESSAGE_LEVEL_INTENTS'
    else:
        intents_or_agents = 'NAME'
    # slim the data set in prep for generating figure
    dfagents = data_file[(data_file['SENTIMENT_TARGET'] == company_or_agent) & (data_file[intents_or_agents] != '[]')]
    dfagents = dfagents[['RAW_SENTIMENT_VALUE', 'CONVERSATION_ID', intents_or_agents]].copy()
    # prep pivot table for figure
    pvt = pd.pivot_table(dfagents, index=intents_or_agents, columns='RAW_SENTIMENT_VALUE', values='CONVERSATION_ID',
                         aggfunc='count')
    pvt.fillna(0, inplace=True)
    pvt['total'] = (pvt[-2] + pvt[-1] + pvt[1] + pvt[2])
    pvt[1] = round((pvt[1] / pvt['total']) * 100, 2)
    pvt[2] = round((pvt[2] / pvt['total']) * 100, 2)
    pvt[-1] = round(((pvt[-1] / pvt['total']) * 100) * -1, 2)
    pvt[-2] = round(((pvt[-2] / pvt['total']) * 100) * -1, 2)
    # pvt[-1] = pvt[-1]*-1
    if order_by > 0:
        pvt.sort_values(by=order_by, ascending=False, inplace=True)
    else:
        pvt.sort_values(by=order_by, inplace=True)
    df = pvt[:top]
    # create the figure
    diverging = go.Figure()
    diverging.add_trace(go.Bar(x=df[-1].values,
                               y=df.index,
                               orientation='h',
                               name=-1,
                               customdata=df[-1],
                               hovertemplate="%{y}: %{customdata}",
                               marker_color='#f86388',
                               text=df[-1] * -1))

    diverging.add_trace(go.Bar(x=df[-2].values,
                               y=df.index,
                               orientation='h',
                               name=-2,
                               customdata=df[-2],
                               hovertemplate="%{y}: %{customdata}",
                               marker_color='#c42f55',
                               text=df[-2] * -1))
    diverging.add_trace(go.Bar(x=df[1],
                               y=df.index,
                               orientation='h',
                               name=1,
                               hovertemplate="%{y}: %{x}",
                               marker_color='#09aea1',
                               text=df[1]))
    diverging.add_trace(go.Bar(x=df[2],
                               y=df.index,
                               orientation='h',
                               name=2,
                               hovertemplate="%{y}: %{x}",
                               marker_color='#077f74',
                               text=df[2]))
    diverging.update_layout(barmode='relative',
                            height=800,
                            width=1400,
                            yaxis_autorange='reversed',
                            bargap=0.3,
                            legend_orientation='v',
                            legend_title_text='Message Sentiment Score',
                            legend_x=1, legend_y=0,
                            plot_bgcolor='#f5f5f5',
                            paper_bgcolor='#fff')

    return diverging
