import re 
import time #time.sleep(1)
import pandas as pd
import numpy as np
from itertools import repeat
import emoji
import warnings
import datetime

#This function will detect the date format
def detect_dt_format(Extract_date):
    lst = Extract_date[0].split('/')
    if(len(lst[2])==4):
        format = '%d/%m/%Y'
    else:
        format = '%d/%m/%y'

    for i in range(len(Extract_date)):
        lst = Extract_date[i].split('/')
        if(int(lst[1])>12): #It is mm/dd/yyyy format 
            if(len(lst[2])==4):
                format = '%m/%d/%Y'
            else:
                format = '%m/%d/%y'
            break
    return format

# This function will convert the date format
def convert(dt,format):
    return datetime.datetime.strptime(dt,format).strftime('%Y-%m-%d')

# Converting 12 hour time to 24 hour time
def to_24hr(time_in_12hr):
    hr_min, am_pm = time_in_12hr.lower().split()
    hrs, mins = [int(i) for i in hr_min.split(":")]
    hrs %= 12
    hrs += 12 if am_pm == 'pm' else 0
    return f"{hrs:02}:{mins:02}:{'00'}"

# Extracting Emojis from the message
def extract_emojis(s):
    return [c for c in s if c in emoji.UNICODE_EMOJI['en']]

# Returning another dataframe for dashboard
def separate_dataframe(df):
    tem = df[['day_name','day_part', 'user', 'message', 'total-words','month']].copy()
    tem = tem[tem['user'] != 'group_notification']
    #tem = tem[tem['message'] != '<Media omitted>']
    #tem = tem[tem['message'] != 'This message was deleted']
    tem = tem[tem['total-words'] > 1]
    time.sleep(0.1)
    
    tem['Date'] = pd.to_datetime(df['only_date'], errors='coerce').dt.strftime('%d-%m-%Y')
    # time.sleep(0.2)
    tem['Time'] = pd.to_datetime(df['date'], errors='coerce').dt.strftime('%H:%M')
    tem['emoji'] = tem["message"].apply(extract_emojis)

    tem.rename(columns={'day_name': 'Day'}, inplace=True)
    tem.rename(columns={'day_part': 'Period'}, inplace=True)
    tem.rename(columns={'user': 'User'}, inplace=True)
    tem.rename(columns={'message': 'Message'}, inplace=True)
    tem.rename(columns={'total-words': 'Total Words'}, inplace=True)
    tem.rename(columns={'month': 'Month'}, inplace=True)
    tem = tem[['Date', 'Time', 'Month', 'Day', 'Period', 'User', 'Message', 'Total Words', 'emoji']]

    tem_df= df[['user', 'message','total-characters','total-words','total-urls','day_name']].copy()
    tem_df['date'] = pd.to_datetime(df['date']).dt.date
    tem_df['time'] = pd.to_datetime(df['date']).dt.time
    tem_df['emojis'] = tem_df["message"].apply(extract_emojis)
    tem_df['media'] = tem_df['message'].apply(lambda x: 1 if '<Media omitted>' in x else 0)
    tem_df['total-urls'].groupby(tem_df['user']).sum()
    tem_df['message_count'] = 1
    tem_df = tem_df.rename(columns={'user': 'name'})
    tem_df = tem_df.rename(columns={'total-urls': 'urlcount'})
    tem_df = tem_df.rename(columns={'total-characters': 'letter_count'})
    tem_df = tem_df.rename(columns={'total-words': 'word_count'})
    tem_df = tem_df.rename(columns={'day_name': 'day'})
    tem_df['time'] = tem_df['time'].apply(lambda x: x.hour*60*60*1000 + x.minute*60*1000 + x.second*1000 + x.microsecond/1000)
    tem_df['time'] = tem_df['time'].astype(int)
    tem_df = tem_df[['name', 'message', 'date', 'time', 'media', 'emojis', 'urlcount', 'letter_count', 'word_count', 'message_count','day']]
    
    return tem, tem_df
    
# Seperate User & Message
def seperate_user_msg(df):
    users = []
    messages = []
    for message in df['user_message']:
        entry = re.split('([\w\W]+?):\s', message)
        if entry[1:]:  # user name
            users.append(entry[1])
            messages.append(" ".join(entry[2:]))
        else:
            users.append('group_notification')
            messages.append(entry[0])
    return users, messages

def preprocess(data):
    """
    Input: 
        dataframe
  
    Output: 
        Processed dataframe
    """
    pattern = '\d{1,2}/\d{1,2}/\d{2,4},\s\d{1,2}:\d{2}\s[AP]M\s-\s|\d{1,2}/\d{1,2}/\d{2,4},\s\d{1,2}:\d{2}\s-\s|\d{1,2}/\d{1,2}/\d{2,4},\s\d{1,2}:\d{2}\s[ap]m\s-\s'

    # Separating Msg and Timestamp
    messages = re.split(pattern, data)[1:]
    timestamp = re.findall(pattern, data)

    # Extracting dates and Times from the timestamp
    pattern2 = '\d{1,2}/\d{1,2}/\d{2,4}'
    Extract_date = re.findall(pattern2,str(timestamp))
    Extract_time = re.findall('\d{1,2}:\d{2}\s[AP]M|\d{1,2}:\d{2}\s[ap]m',str(timestamp))
    time.sleep(0.3)

    # Detecting the date format
    format = detect_dt_format(Extract_date)

    # Converting date to YYYY-MM-DD format and times to 24 hour format
    Extract_date = list(map(convert,Extract_date,repeat(format)))
    Extract_time = list(map(to_24hr,Extract_time))
    final_date_time = [ x + ' ' + y for x,y in zip(Extract_date,Extract_time)]
    
    # Preparing the dataframe
    l1 = list(messages)
    l2 = list(final_date_time)
    s1 = pd.Series(l1, name='user_message')
    s2 = pd.Series(l2, name='date')
    df = pd.concat([s1,s2], axis=1)
    time.sleep(0.3)

    # Seperating user and message
    users, messages = seperate_user_msg(df)
    df['user'] = users
    df['message'] = messages
    df.drop(columns=['user_message'], inplace=True)

    # Extracting Day and Time Details
    df['date'] = pd.to_datetime(df.date, format='%Y-%m-%d %H:%M:%S')
    time.sleep(0.3)

    df['only_date'] = df['date'].dt.date
    df['year'] = df['date'].dt.year
    df['month_num'] = df['date'].dt.month
    df['month'] = df['date'].dt.month_name()
    df['day'] = df['date'].dt.day
    df['day_name'] = df['date'].dt.day_name()
    df['hour'] = df['date'].dt.hour
    df['minute'] = df['date'].dt.minute

    # Converting Given Time to Day_Time
    df['day_part'] = (df['hour']% 24 + 4) // 4
    df['day_part'].replace({1: 'Late Night',
                      2: 'Early Morning',
                      3: 'Morning',
                      4: 'Noon',
                      5: 'Evening',
                      6: 'Night'}, inplace=True)

    # Defining the Period                 
    period = []
    for hour in df[['day_name', 'hour']]['hour']:
        if hour == 23:
            period.append(str(hour) + "-" + str('00'))
        elif hour == 0:
            period.append(str('00') + "-" + str(hour + 1))
        else:
            period.append(str(hour) + "-" + str(hour + 1))
    
    df['period'] = period

    # Cleaning the dataframe
    df = df[df['user'] != 'group_notification']
    df['message'] = df['message'].str.replace('\n', '')
    time.sleep(0.3)

    ### Statistics of the dataframe
    df['total-words'] = df['message'].str.split().str.len()
    df['total-characters'] = df['message'].apply(lambda s : len(s))
    df['total-unique-words'] = [len(set(x.split())) for x in df['message']]
    df['total-urls'] = df['message'].str.count('http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\(\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+')
    time.sleep(0.2)
    
    df2,df3 = separate_dataframe(df)
    
    ### Function to count number of media in chat.
    MEDIAPATTERN = r'<Media omitted>'
    df['Media_Count'] = df.message.apply(lambda x : re.findall(MEDIAPATTERN, x)).str.len()
    df['total-emojis'] = df['message'].str.count('[\U0001F600-\U0001F64F]')
    df['total-emojis-no-numbers'] = df['message'].str.count('[\U0001F600-\U0001F64F][^0-9]')
    df['total-punctuation'] = df['message'].str.count('[^\w\s]')
    df['data_type'] = df['message'].str.count('[^\w\s]').apply(lambda x: 'punctuation' if x > 0 else 'text')
    df['isnumeric'] = df['message'].str.isnumeric()

    return df,df2,df3