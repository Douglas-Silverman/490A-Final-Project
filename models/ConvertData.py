import pandas as pd
import re



def convert_data(file_name):
    data_struct = []
    data = pd.read_csv(file_name, encoding = 'latin-1')
    for index, row in data.iterrows():
        tweet = row["OriginalTweet"]
        tweet = clean_data(tweet)
        sentiment = row["Sentiment"]
        data_struct.append([tweet, sentiment])
    
    return data_struct


# print(pd.read_csv("./Datasets/Corona_NLP_train.csv", encoding= ))


def clean_data(string):


    # tweet_array = convert_data(file_name)
    # for elem in tweet_array:
    #     for char in tweet_array[0]:
    text = string
    text = text.lower()
    text = re.sub(r'http\S+', '', text)

    punc = '''!()[]{};:'"\,<>./$^*_~'''
    for ele in text:  
        if ele in punc:  
            text = text.replace(ele, "")

    regrex_pattern = re.compile(pattern = "["
        u"\U0001F600-\U0001F64F"  # emoticons
        u"\U0001F300-\U0001F5FF"  # symbols & pictographs
        u"\U0001F680-\U0001F6FF"  # transport & map symbols
        u"\U0001F1E0-\U0001F1FF"  # flags (iOS)
                           "]+", flags = re.UNICODE)

    return regrex_pattern.sub(r'',text)

clean_data("Why dont we make most grocery stores pickup only for the next few weeks? Would reduce crowding, panic buying, contamination, strain on workers. @Google could help build an app in a day. Open each store for 2 hrs for seniors who might be less tech savvy. #COVID2019 #Coronavirus")
text = u'This is a smiley face \U0001f602'
print(text) # with emoji
print(clean_data(text))


def get_tweet(file_name):
    return convert_data(file_name)[0]


def get_sentiment(file_name):
    return convert_data(file_name)[1]