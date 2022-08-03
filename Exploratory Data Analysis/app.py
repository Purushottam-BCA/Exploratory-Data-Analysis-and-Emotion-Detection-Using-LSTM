# Libraries Imported
import time
import os
import re
from wordcloud import WordCloud, STOPWORDS
import pandas as pd
import streamlit as st
from torch import t
import preprocessor,helper
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
from st_aggrid import AgGrid, GridOptionsBuilder
from st_aggrid.shared import GridUpdateMode
from helper import emoji_extract,most_common_words,cloud_data,create_wordcloud,pie_display_emojis, fetch_url_list, time_series_plot,day_wise_pie, message_cluster, most_active_member, most_busy_day,\
    most_busy_month,most_busy_hour,url_scatter_bubble, max_words_used, url_bar_graph, top_media_contributor, who_shared_links,\
    sentiment_analysis,top_10_days,day_wise_count,url_line_graph

from custom_modules import func_analysis as analysis

# to disable warning by file_uploader going to convert into io.TextIOWrapper
st.set_option('deprecation.showfileUploaderEncoding', False)
st.set_option('deprecation.showPyplotGlobalUse', False)

warnings.filterwarnings("ignore", message="Glyph 128584 missing from current font.")

def aggrid_interactive_table(df: pd.DataFrame):
    options = GridOptionsBuilder.from_dataframe(
        df, enableRowGroup=True, enableValue=True, enablePivot=True
    )

    options.configure_side_bar()

    options.configure_selection("single")
    selection = AgGrid(
        df,
        enable_enterprise_modules=True,
        gridOptions=options.build(),
        theme="light",
        update_mode=GridUpdateMode.MODEL_CHANGED,
        allow_unsafe_jscode=True,
    )

    return selection

def add_multilingual_stopwords():
    """
    Function read language file stop words and convert
    them into List of STOPWORDS.
    Top languages added under stopwords folder.

    attributes
    ----------
    None

    Returns
    -------
    set: Distinct list of words
    """
    multilingul_list = []
    for file in os.listdir('configs/stopwords'):
        stopword = open('configs/stopwords/' + file, "r",encoding="utf-8")
        for word in stopword:
            word = re.sub('[\n]', '', word)
            multilingul_list.append(word)
    return set(STOPWORDS).union(set(multilingul_list))

def generate_word_cloud(text: str):
    """
    Function takes text as input and transform it to
    WordCloud display

    attributes
    ----------
    text (str): String of words
    title (str): title Sting

    Return
    ------
    Matplotlib figure for wordcloud
    """
    wordcloud = WordCloud(
        scale=3,
        width=650,
        height=330,
        max_words=200,
        colormap='viridis', #'tab20c',
        stopwords=add_multilingual_stopwords(),
        collocations=True,
        contour_color='#5d0f24',
        contour_width=3,
        font_path='Laila-Regular.ttf',
        background_color="white").generate(text)

    plt.figure(figsize=(10, 8))
    plt.imshow(wordcloud, interpolation='bilinear')
    plt.axis("off")
    st.pyplot()

# Displaying the stats
def display_stats(stats):
    st.sidebar.text("")
    st.header("♟ General Statistics ♟")
    col1, col2, col3, col4, col5, col6 = st.columns(6)

    col1.metric(
    "Total Messages", stats.get('total_messages'), delta="📦 💬")
    col2.metric(
    "Total Words", stats.get('total_words'), delta="📜 📜")
    col3.metric(
    "Total Members", stats.get('total_members'),delta= "💃🕺")
    col4.metric(
    "Total Media", stats.get('media_message'), delta="🎞️ 📷")
    col5.metric(
    "Link shared", stats.get('link_shared'), delta="🖇️ 🔗")
    col6.metric(
    "Deleted Messages", stats.get('total_deleted_messages'), delta="🗑️ ♻️")
    
    st.text("")
    st.write("")

    col1, col2, col3, col4, col5, col6 = st.columns(6)

    col1.metric(
    "Total Emojis", stats.get('total_emojis'), delta="🙂 😎")
    col2.metric(
    "First Msg Date", stats.get('first_date'), delta="📅 📆")
    col3.metric(
    "Last Msg Date", stats.get('last_date'),delta= "📆 📅")
    col4.metric(
    "Average Word/Msg", stats.get('avg_word_length'), delta="✉ 💬")
    col5.metric(
    "Most Active Day", stats.get('most_active_day'), delta="📰 🚀")
    col6.metric(
    "Least Active Day", stats.get('least_active_day'), delta="⌚ 💡")
    st.text("")

# Displaying Charts
def chart_display(df2,df3,temp_df,strn):
    
    # Over The Time Analysis
    st.markdown("----")
    plt.set_loglevel('WARNING') 
    st.header("🔘 Over the Time Analysis ")
    st.info("🔋 Analysis of number of messages using Time Series plot.")
    st.write(time_series_plot(df2))
    time.sleep(0.2)

    # Busy Month and Busy Day
    col1,col2 = st.columns(2)
    with col1:
        st.markdown("----")
        st.subheader("🔘 Most active months during the chat")
        st.info("🔋 Comparision on the number of messages posted in different Months")
        st.pyplot(most_busy_month(df2))
    with col2:
        st.markdown("----")
        st.subheader("🔘 Most active Days during the chat")
        st.info("🔋 Comparision on the number of messages posted in different Days")
        st.pyplot(most_busy_day(df2)) 
    time.sleep(0.2)
    
    # Busy Hour (24 Hour Clock)    
    st.markdown("----")
    st.header("🔘 Most busy hours")
    st.info("🔋 Comparision on the number of messages posted in each hour")
    fig, ax = most_busy_hour(df2)
    sns.set_style("darkgrid")
    st.pyplot(fig) 
    time.sleep(0.1)

    # Week and Hour HeatMap
    st.markdown("----")
    st.header("🔘 Group highly Active time during Weeks and Hours")
    st.info("🔋 Comparision on messages posted in weekday and hour wise")
    user_heatmap = helper.activity_heatmap_week(df2)
    fig,ax = plt.subplots()
    sns.set_style("darkgrid")
    sns.heatmap(user_heatmap,cmap="YlGnBu",annot=True,fmt='.0f',ax=ax)
    st.pyplot(fig)
    time.sleep(0.1)
    
    # Month and Week HeatMap
    st.markdown("----")
    st.header("🔘 Group highly Active time with respect to Month and Weeks")
    st.info("🔋 Comparision on messages posted in Month and Week wise")
    sns.set_style("darkgrid")
    user_heatmap = helper.activity_heatmap_month(df3)
    fig,ax = plt.subplots()
    sns.heatmap(user_heatmap,cmap='coolwarm',annot=True,fmt='.0f',ax=ax)
    st.pyplot(fig)
    time.sleep(0.1)
    
    # Top 10 Members 
    st.markdown("----")
    st.subheader("🔘 Most Active Members in the Chat")
    order = ['Top 15 Users','Top 10 Users','Top 20 Users','Top 25 Users','Top 30 Users','Top 35 Users','Top 40 Users','Top 45 Users','Top 50 Users']
    num = st.selectbox("Select the number of users to be displayed",order)
    n = int(num[4:6])
    n=15
    st.info("🔋 Member comparision based on the number of messages posted in chat")
    st.pyplot(most_active_member(df3,n))
    
    st.markdown("----")
    st.header("🔘 Who uses more words in sentences")
    st.info("🔋 Member uses more number of sentences during the conversation")
    st.pyplot(max_words_used(df3))
    time.sleep(0.2)

    col1,col2 = st.columns(2)
    with col1:
        st.markdown("----")
        st.subheader("🔘 Top 10 Most active Days")
        st.info("🔋 Comparision on the number of messages posted in top 10 Dates")
        st.pyplot(top_10_days(df3)) 
    with col2:
        st.markdown("----")
        st.subheader("🔘 Who shares most Links in group ?")
        st.info("🔋 Members who shares internet links of information with others")
        st.pyplot(who_shared_links(df3))

    st.markdown("----")
    st.header("🔘 Top-15 Media Contributors ")
    st.info("🔋 Comparision of members who contributes more number of Images, Video or Documents")
    st.pyplot(top_media_contributor(df3))
    time.sleep(0.2)

    st.markdown("----")
    st.header("🔘 Day Wise Distribution of Messages")
    st.info("🔋 Comparision of number of messages posted in different Days")
    col1,col2 = st.columns(2)
    with col1:
        st.plotly_chart(day_wise_count(df))
    with col2:
        st.plotly_chart(day_wise_pie(df3))  
    
    # URL Analysis
    st.markdown("----")
    st.subheader("🔘 Most Shared URLs")
    st.info("🔋 Comparision on the number of Websites shared")
    col1,col2,col3 = st.columns([2,1,1])
    urlMain, urlList,urlList2 = fetch_url_list(df['message'])
    with col1:
        st.subheader("🔘 Columner Chart")
        st.pyplot(url_bar_graph(urlList))
    with col2:
        st.write(urlList)
    with col3:
        st.write(urlList2)
    time.sleep(0.1)

    st.markdown("----")
    st.subheader("🔘 Most Shared URLs [ Websites ]")
    st.info("🔋 Comparision on the number of Websites shared")
    st.write(url_line_graph(urlMain))
    st.write(url_scatter_bubble(urlMain))
    time.sleep(0.1)

    st.markdown("----")
    st.subheader("🔘 Most Common Words Used During Chat")
    st.info("🔋 Comparision on the number of times each word is used in chat")
    most_common_df = most_common_words(temp_df)
    fig, ax = plt.subplots(figsize=(12, 5))
    ax.bar(most_common_df[0], most_common_df[1], color='orange')
    ax.set_title('Top 25 used words in chat among all the members')
    ax.set_xlabel('Used Words in the conversation')
    ax.set_ylabel('Occurance of words')
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    plt.setp(ax.get_xticklabels(), rotation=45, ha="right", rotation_mode="anchor")
    for i, v in enumerate(most_common_df[1]):
        ax.text(i, v, str(v), color='blue', fontsize=8, ha='center', va='bottom', rotation=45, bbox=dict(boxstyle='circle,pad=0.3', fc='white', alpha=0.5))
    st.pyplot(fig)

    st.markdown("----")
    st.subheader("🔘 Word Cloud for Words and Phrases frequently used in Chat")
    st.info("🔋 Frequently used words or phrases by all members in group chat. Most dicussion occurs around below words or used frequently.")
    text = create_wordcloud(df3)
    generate_word_cloud(text)
    
    st.markdown("----")
    st.header("🔘 Clustering on the basis of Member activity")
    st.info("🔋 Cluster hover about the total messages, Emoji's, Links, Words\
        and Letter by individual member")
    st.write(message_cluster(temp_df))
    

    st.markdown("----")
    st.header("🔘 Curious about Emoji's ?")
    st.info("🔋 The most use Emoji's in converstion is show with larger sector")
    col1,col2 = st.columns(2)
    d1, d2 = emoji_extract(df2)
    with col1:
        st.dataframe(d1)
    with col2:
        st.dataframe(d2)

    st.markdown("----")
    st.header("🔘 Emoji's used by each member")
    st.info("🔋 Emoji's used by each member in group chat")
    pie_display = pie_display_emojis(strn)
    st.plotly_chart(pie_display)
    
def positive_sentiment(df):
    st.markdown("----")
    st.subheader("🔘 Who has Positive Sentiment ?")
    st.info("🔋 Member sentiment analysis score base on the words used in\
        messages. Sentiment Score above 0.4 to 1 is consider as Positive.\
            Pure English words and Phrases is ideal for calculation")
    st.pyplot(sentiment_analysis(df))


# Initial page config
st.set_page_config(
    page_title='Exploratory Data Analysis on Whatsapp',
    page_icon="",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.title("Exploratory Data Analysis on Whatsapp")
st.markdown("To analyze the WhatsApp Chat using the exported text file 📁.")

# CSS Code
st.markdown("""
    <style>
    
    .css-1xarl3l {
        font-size: 1.9rem;
        padding-bottom: 0.25rem;
    }

    .css-18e3th9 {
        padding-left: 4rem;
        padding-right: 4rem;
        padding-top: 2rem;
    }

    .css-hxt7ib {
        padding-top: 2.5rem;
        padding-left: 1rem;
        padding-right: 1rem;
        padding-bottom: 1.5rem;
    }

    .css-pmxsec .exg6vvm10 {
        margin-top: 6px;
        padding-right: 1rem;
        border: outset;
        border-width: 1px;
        border-color: #dee2e6;
    }

    .css-1a32fsj.e19lei0e0 {
        margin-left: -22px;
    }
    .stPlotlyChart.js-plotly-plot {
        margin-left: -40px;
        margin-top: -15px;
    }

   .css-pmxsec .exg6vvm15 {
        border: ridge;
        display: flex;
        flex-direction: column;
        align-items: flex-start;
    },
}

     </style>
 """,unsafe_allow_html=True,)

# Sidebar 
st.sidebar.title("Instructions")
st.sidebar.markdown('**STEP 1 : Exporting chat file**')
st.sidebar.text('◾ Open Whatsapp.')
st.sidebar.text('◾ Open any individual or group chat.')
st.sidebar.text('◾ Tap options [...] > Export chat.')
st.sidebar.text('◾ Choose export with/without media.')

st.sidebar.markdown('**STEP 2 : Upload the chat file**')
uploaded_file = st.sidebar.file_uploader("◾ Choose the chat file (txt) : ",type=["txt"], accept_multiple_files=False)
if uploaded_file is not None:
    # Convert txt string to utf-8 Encoding
    data = uploaded_file.getvalue().decode("utf-8")
    
    # Preprocessing
    df, df2, temp = preprocessor.preprocess(data)
    df3 = df2.copy()
    df4 = df2.copy()
    time.sleep(0.2)

    # Formation of Word Cloud Dataframe
    cloud_df = cloud_data(temp)
    if st.sidebar.checkbox("Show chat data ", False):
        st.subheader("📊 Conversation Data : (Normal Table) ##")
        st.write(df2)

    if st.sidebar.checkbox("Show interactive table ", False):
        st.subheader("📊 Conversation Data : (Interactive Table) ")
        selection = aggrid_interactive_table(df=df2)

        if selection:
            st.write(" You selected : ")
            st.json(selection["selected_rows"])
            time.sleep(0.3)
    
    st.sidebar.markdown("**STEP 3 : Select User to Analyze**")
    user_list = analysis.get_user_list(df)

    selected_user = st.sidebar.selectbox(" Member Names 👇👇 ",user_list)
    if st.sidebar.button("Show Analysis"):

        if selected_user != 'All':
            df2 = df[df['user'] == selected_user]
        else:
            df2 = df

        stats = helper.fetch_stats(df2)

        display_stats(stats)

        chart_display(df2,df3, temp,df4)
        
        positive_sentiment(cloud_df)

st.sidebar.markdown('----')