import pandas as pd
import numpy as np
import streamlit as st
import qbr_functions as qbr
import plotly.express as px

st.set_page_config(layout='wide')

# First - define the data set we are working with. FUTURE: this will be an argument passed from the terminal by the user
# using arg parse

# '/Users/nickluzio/PycharmProjects/Auctane QBR/auctane QA (4).csv'
# C:\Users\Nick Luzio\Downloads\auctane QA (4).csv


# Building the streamlit app features
# title of the app
st.title('qbr dashboard ')

# get the data and load it
if st.checkbox('Use your own data?'):
    data_file = st.file_uploader('Choose your data file. Ignore the error, it will go away once you upload a file. I '
                                 'will fix it soon. :)')
    data = qbr.data_loader(data_file)
    clean_data = data.shape[0]
else:
    st.write('*REMINDER* This is randomly generated data. To use your own dataset, use the checkbox above')
    data_file = pd.DataFrame(np.random.randint(300, 10000, size=(5000, 7)), columns=['CONVERSATION_ID', 'CONV_ART',
                                                                                     'CONV_FRT', 'CONV_DURATION',
                                                                                     'CSAT_SCORE',
                                                                                     'AGENT_NAME', 'usage'])
    rand_dates = pd.DataFrame(
        qbr.random_dates(start=pd.to_datetime('2022-07-01'), end=pd.to_datetime('2022-10-01'), n=5000))
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
        scatter = px.scatter(data, x='usage', y='CONV_ART', width=500, marginal_y='violin', marginal_x='histogram')
        st.plotly_chart(scatter)
    with FRT:
        st.subheader('Distribution of FRT')
        scatter2 = px.scatter(data, x='usage', y='CONV_FRT', width=500, marginal_y='violin', marginal_x='histogram')
        st.plotly_chart(scatter2)
    with Duration:
        st.subheader('Distribution of Duration')
        scatter3 = px.scatter(data, x='usage', y='CONV_DURATION', width=600, marginal_y='violin', marginal_x='histogram')
        st.plotly_chart(scatter3)
if st.checkbox('Usage Based Analysis'):
    with st.container():
        st.header('High-Low Use Analysis')
        usage = st.slider('Usage %')
        st.write('Median Usage = ', data['usage'].median())
        period = st.text_input(
            'Period for Analysis. Use M or W, the default period start is Sunday. You can specify the '
            'day you want your grouping to start '
            'on by using w-MON, w-TUE.', value='W')
        high_low = qbr.usage_spliter(data, usage)
        st.write('Date Range = ', data['date'].min(), ' - ', data['date'].max())
        selected_teams = st.checkbox('Use Teams?')
        if selected_teams:
            check_period = st.checkbox('Use Period Analysis?')
        with st.container():
            selection = st.multiselect('Select Which Metrics To Run:',
                                       ['CONV_ART', 'CONV_FRT', 'CONV_DURATION', 'CSAT_SCORE'])
            st.subheader('Average Based Change Analysis')
            High, Low = st.columns(2)
            with High:
                high_data = high_low[0]
                st.subheader('High Usage Statistics')
                st.write('High Use Conversation Count = ', high_data.shape[0])
                csat_h = (high_data['CSAT_SCORE'].shape[0] - high_data['CSAT_SCORE'].isna().sum())/ high_data.shape[0]
                st.write('CSAT Response Rate = ', round(csat_h*100, 2), '%')
                if selected_teams:
                    if check_period:
                        high_stats = high_data.groupby(
                            [high_data['TEAM_NAME'], pd.Grouper(key='date', freq=period)])[
                            selection].mean().reset_index().set_index('TEAM_NAME')
                    else:
                        high_stats = high_data.groupby(high_data['TEAM_NAME'])[selection].mean().reset_index(
                        ).set_index('TEAM_NAME')
                else:
                    high_stats = high_data.groupby(pd.Grouper(key='date', freq=period))[
                        selection].mean().reset_index().set_index('date')
                st.table(high_stats)
                val_high = high_data[selection].mean().reset_index().set_index('index')
                val_high.rename(columns={0: 'Overall Average'}, inplace=True)
                st.table(val_high)
            with Low:
                low_data = high_low[1]
                st.subheader('Low Usage Statistics')
                st.write('Low Use Conversation Count = ', low_data.shape[0])
                csat_l = (low_data['CSAT_SCORE'].shape[0] - low_data['CSAT_SCORE'].isna().sum()) / low_data.shape[0]
                st.write('CSAT Response Rate = ', round(csat_l * 100, 2), '%')
                if selected_teams:
                    if check_period:
                        low_stats = low_data.groupby(
                            [low_data['TEAM_NAME'], pd.Grouper(key='date', freq=period)])[
                            selection].mean().reset_index().set_index('TEAM_NAME')
                    else:
                        low_stats = low_data.groupby(low_data['TEAM_NAME'])[selection].mean().reset_index().set_index(
                                    'TEAM_NAME')
                else:
                    low_stats = low_data.groupby(pd.Grouper(key='date', freq=period))[
                        selection].mean().reset_index().set_index('date')
                st.table(low_stats)
                val_low = low_data[selection].mean().reset_index().set_index('index')
                val_low.rename(columns={0: 'Overall Average'}, inplace=True)
                st.table(val_low)
            st.subheader('Percent Change')
            CONV_ART, CONV_FRT, CONV_DURATION, CSAT_SCORE = st.columns(4)
            with CONV_ART:
                delta = ((high_data['CONV_ART'].mean() - low_data['CONV_ART'].mean()) / low_data[
                    'CONV_ART'].mean()) * 100
                qbr.make_metric(name='Average ART', val=high_data['CONV_ART'].mean(), change=delta, d_color='inverse')
            with CONV_FRT:
                delta = ((high_data['CONV_FRT'].mean() - low_data['CONV_FRT'].mean()) / low_data[
                    'CONV_FRT'].mean()) * 100
                qbr.make_metric(name='Average FRT', val=high_data['CONV_FRT'].mean(), change=delta, d_color='inverse')
            with CONV_DURATION:
                delta = ((high_data['CONV_DURATION'].mean() - low_data['CONV_DURATION'].mean()) / low_data[
                    'CONV_DURATION'].mean()) * 100
                qbr.make_metric(name='Average Duration', val=high_data['CONV_DURATION'].mean(), change=delta,
                                d_color='inverse')
            with CSAT_SCORE:
                delta = ((high_data['CSAT_SCORE'].mean() - low_data['CSAT_SCORE'].mean()) / low_data[
                    'CSAT_SCORE'].mean()) * 100
                qbr.make_metric(name='Average CSAT', val=high_data['CSAT_SCORE'].mean(), change=delta, d_color='normal')

if st.checkbox('Date Based Analysis'):
    with st.container():
        st.header('Pre-Post Analysis')
        date_break = st.date_input('Date for Pre-Post Analysis. Make sure your period grouping matches the day of your '
                                   'break for cleanest results. If your break is a monday, make your period start on '
                                   'monday too', value=pd.to_datetime('2022-08-14'))
        date_break = pd.to_datetime(date_break)
        period = st.text_input(
            'Period for Analysis. Use M or W, the default period start is Sunday. You can specify the '
            'day you want your grouping to start '
            'on by using w-MON, w-TUE.', value='W')
        pre_post = qbr.date_splitter(data, date_break)
        st.write('Date Range = ', data['date'].min(), ' - ', data['date'].max())
        with st.container():
            selection = st.multiselect('Select Which Metrics To Run:',
                                       ['CONV_ART', 'CONV_FRT', 'CONV_DURATION', 'CSAT_SCORE'])
            st.subheader('Average Based Change Analysis')
            Pre, Post = st.columns(2)
            with Pre:
                pre_data = pre_post[0]
                st.subheader('Pre-Date Statistics')
                pre_stats = pre_data.groupby(pd.Grouper(key='date', freq=period))[
                    selection].mean().reset_index().set_index(
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
                delta = ((post_data['CONV_ART'].mean() - pre_data['CONV_ART'].mean()) / pre_data[
                    'CONV_ART'].mean()) * 100
                qbr.make_metric(name='Average ART', val=post_data['CONV_ART'].mean(), change=delta, d_color='inverse')
            with CONV_FRT:
                delta = ((post_data['CONV_FRT'].mean() - pre_data['CONV_FRT'].mean()) / pre_data[
                    'CONV_FRT'].mean()) * 100
                qbr.make_metric(name='Average FRT', val=post_data['CONV_FRT'].mean(), change=delta, d_color='inverse')
            with CONV_DURATION:
                delta = ((post_data['CONV_DURATION'].mean() - pre_data['CONV_DURATION'].mean()) / pre_data[
                    'CONV_DURATION'].mean()) * 100
                qbr.make_metric(name='Average Duration', val=post_data['CONV_DURATION'].mean(), change=delta,
                                d_color='inverse')
            with CSAT_SCORE:
                delta = ((post_data['CSAT_SCORE'].mean() - pre_data['CSAT_SCORE'].mean()) / pre_data[
                    'CSAT_SCORE'].mean()) * 100
                qbr.make_metric(name='Average CSAT', val=post_data['CSAT_SCORE'].mean(), change=delta, d_color='normal')

with st.container():
    st.header('Sentiment Target Classification')
    stc_file = st.file_uploader('Upload STC File')
    stc_file = pd.read_csv(stc_file)
    company_agent, intents_agent, order, top = st.columns(4)
    with company_agent:
        stc_target = st.selectbox('Sentiment Target?', ['Company', 'Agent'])
        stc_target = stc_target.lower()
    with intents_agent:
        group_by = st.selectbox('Group By Agent or Intent?', ['Intents', 'Agents'])
    with order:
        order_it = st.selectbox('Order By?', [-2, -1, 1, 2])
    with top:
        cut_them = st.number_input('Number of results', value=10)
    st.plotly_chart(qbr.diverging_sentiment(data_file=stc_file, company_or_agent=stc_target, intents_or_agents=group_by,
                                            order_by=order_it, top=cut_them))
