import pandas as pd
import numpy as np
import streamlit as st
import plotly.express as px

st.set_page_config(layout='wide')


# First - define the data set we are working with. FUTURE: this will be an argument passed from the terminal by the user
# using arg parse

# '/Users/nickluzio/PycharmProjects/Auctane QBR/auctane QA (4).csv'
# C:\Users\Nick Luzio\Downloads\auctane QA (4).csv


# define some functions
@st.cache
def data_loader(first_data):
    data = pd.read_csv(first_data)
    data['date'] = pd.to_datetime(data['CONV_CREATED'])
    return data


@st.cache
def make_str(text):
    txt = str(text)
    return txt


@st.cache
def date_splitter(data_set, splitter):
    splitter1 = pd.to_datetime(splitter)
    pre = data_set[data_set['date'] < splitter1].copy()
    post = data_set[data_set['date'] > splitter1].copy()
    dset = [pre, post]
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


# Building the streamlit app features
# title of the app
st.title('qbr dashboard ')

# get the data and load it
if st.checkbox('Use your own data?'):
    data_file = st.file_uploader('Choose your data file. Ignore the error, it will go away once you upload a file. I '
                                 'will fix it soon. :)')
    data = data_loader(data_file)
    clean_data = data.shape[0]
else:
    st.write('*REMINDER* This is randomly generated data. To use your own dataset, use the checkbox above')
    data_file = pd.DataFrame(np.random.randint(300, 10000, size=(5000, 6)), columns=['CONVERSATION_ID', 'CONV_ART',
                                                                                     'CONV_FRT', 'CONV_DURATION',
                                                                                     'CSAT_SCORE',
                                                                                     'AGENT_NAME'])
    rand_dates = pd.DataFrame(
        random_dates(start=pd.to_datetime('2022-07-01'), end=pd.to_datetime('2022-10-01'), n=5000))
    data = rand_dates.join(data_file)
    data.rename(columns={0: 'date'}, inplace=True)
    clean_data = data.shape[0]

# sidebar build
with st.sidebar:
    trimmed_convos = 0.0
    if st.checkbox('Trim Outliers?'):
        art_trimmer = st.number_input('Trim ART Samples (seconds):')
        frt_trimmer = st.number_input('Trim FRT Samples (seconds):')
        dur_trimmer = st.number_input('Trim Duration Samples (seconds):')
        if art_trimmer != 0.0:
            art_data = data[data['CONV_ART'] <= art_trimmer]
            trimmed_art = data[data['CONV_ART'] > art_trimmer].shape[0]
            trimmed_convos = trimmed_convos + trimmed_art
            data = art_data.copy()
        if frt_trimmer != 0.0:
            frt_data = data[data['CONV_FRT'] <= frt_trimmer]
            trimmed_frt = data[data['CONV_FRT'] > frt_trimmer].shape[0]
            trimmed_convos = trimmed_convos + trimmed_frt
            data = frt_data.copy()
        if dur_trimmer != 0.0:
            dur_data = data[data['CONV_DURATION'] <= dur_trimmer]
            trimmed_dur = data[data['CONV_DURATION'] > dur_trimmer].shape[0]
            trimmed_convos = trimmed_convos + trimmed_dur
            data = dur_data.copy()

    usage_breakpoint = st.number_input('Usage % Slicer:')

# Body of app


# raw data if user needs it
if st.checkbox('Show raw data'):
    st.subheader('Raw Data')
    st.write(data)

# chat volume checker
if st.checkbox('Show Chats Per Agent'):
    st.subheader('Chat Volume')
    convos = data.groupby(data['date'].dt.to_period('w'))['CONVERSATION_ID'].count().reset_index()
    agents = data.groupby(data['date'].dt.to_period('w'))['AGENT_NAME'].nunique().reset_index()
    line = convos.merge(agents, on='date')
    line['Chats Per Agent'] = line['CONVERSATION_ID'] / line['AGENT_NAME']
    st.line_chart(line, x='date', y='Chats Per Agent')

# Show distributions of major metrics
with st.container():
    st.header('Distributions')
    st.write('Original Conversation Count = ', clean_data)
    st.write('Number of outliers cut = ', trimmed_convos)
    p_removed = trimmed_convos / clean_data
    st.write('Percent of Conversations removed =', p_removed)
    ART, FRT, Duration = st.columns(3, gap='large')
    with ART:
        st.subheader('Distribution of ART')
        scatter = px.scatter(data, x='CONVERSATION_ID', y='CONV_ART', width=600)
        st.plotly_chart(scatter)
    with FRT:
        st.subheader('Distribution of FRT')
        scatter2 = px.scatter(data, x='CONVERSATION_ID', y='CONV_FRT', width=600)
        st.plotly_chart(scatter2)
    with Duration:
        st.subheader('Distribution of Duration')
        scatter3 = px.scatter(data, x='CONVERSATION_ID', y='CONV_DURATION', width=600)
        st.plotly_chart(scatter3)

with st.container():
    st.header('Pre-Post Analysis')
    date_break = st.date_input('Date for Pre-Post Analysis. Make sure your period grouping matches the day of your '
                               'break for cleanest results. If your break is a monday, make your period start on '
                               'monday too', value=pd.to_datetime('2022-08-14'))
    date_break = pd.to_datetime(date_break)
    period = st.text_input('Period for Analysis. Use M or W, the default period start is Sunday. You can specify the '
                           'day you want your grouping to start '
                           'on by using w-MON, w-TUE.', value='W')
    pre_post = date_splitter(data, date_break)
    st.write('Date Range = ', data['date'].min(), ' - ', data['date'].max())
    with st.container():
        selection = st.multiselect('Select Which Metrics To Run:',
                                   ['CONV_ART', 'CONV_FRT', 'CONV_DURATION', 'CSAT_SCORE'])
        st.subheader('Average Based Change Analysis')
        Pre, Post = st.columns(2)
        with Pre:
            pre_data = pre_post[0]
            st.subheader('Pre-Date Statistics')
            pre_stats = pre_data.groupby(pd.Grouper(key='date', freq=period))[selection].mean().reset_index().set_index(
                'date')
            st.table(pre_stats)
            val_pre = pre_data[selection].mean().reset_index().set_index('index')
            val_pre.rename(columns={0: 'Overall Average'}, inplace=True)
            st.table(val_pre)
        with Post:
            post_data = pre_post[1]
            st.subheader('Post-Date Statistics')
            post_stats = post_data.groupby(pd.Grouper(key='date', freq=period))[
                selection].mean().reset_index().set_index('date')
            st.table(post_stats)
            val_post = post_data[selection].mean().reset_index().set_index('index')
            val_post.rename(columns={0: 'Overall Average'}, inplace=True)
            st.table(val_post)
        st.subheader('Percent Change')
        CONV_ART, CONV_FRT, CONV_DURATION, CSAT_SCORE = st.columns(4)
        with CONV_ART:
            delta = ((post_data['CONV_ART'].mean() - pre_data['CONV_ART'].mean()) / pre_data['CONV_ART'].mean()) * 100
            make_metric(name='Average ART', val=post_data['CONV_ART'].mean(), change=delta, d_color='inverse')
        with CONV_FRT:
            delta = ((post_data['CONV_FRT'].mean() - pre_data['CONV_FRT'].mean()) / pre_data['CONV_FRT'].mean()) * 100
            make_metric(name='Average FRT', val=post_data['CONV_FRT'].mean(), change=delta, d_color='inverse')
        with CONV_DURATION:
            delta = ((post_data['CONV_DURATION'].mean() - pre_data['CONV_DURATION'].mean()) / pre_data['CONV_DURATION'].mean()) * 100
            make_metric(name='Average Duration', val=post_data['CONV_DURATION'].mean(), change=delta, d_color='inverse')
        with CSAT_SCORE:
            delta = ((post_data['CSAT_SCORE'].mean() - pre_data['CSAT_SCORE'].mean()) / pre_data['CSAT_SCORE'].mean()) * 100
            make_metric(name='Average CSAT', val=post_data['CSAT_SCORE'].mean(), change=delta, d_color='normal')


with st.container():
    st.header('Sentiment Target Classification')
