import pandas as pd
import re


def clean_data(string):

    # tweet_array = convert_data(file_name)
    # for elem in tweet_array:
    #     for char in tweet_array[0]:
    text = string
    text = text.lower()
    text = re.sub(r'http\S+', '', text)

    allowed_punc = '''abcdefghijklmnopqrstuvwxyz-@$#%0123456789 '''
    for ele in text:  
        if ele not in allowed_punc:  
            text = text.replace(ele, "")


    # https://stackoverflow.com/questions/33404752/removing-emojis-from-a-string-in-python
    regrex_pattern = re.compile(pattern = "["
        u"\U0001F600-\U0001F64F"  # emoticons
        u"\U0001F300-\U0001F5FF"  # symbols & pictographs
        u"\U0001F680-\U0001F6FF"  # transport & map symbols
        u"\U0001F1E0-\U0001F1FF"  # flags (iOS)
                           "]+", flags = re.UNICODE)

    return regrex_pattern.sub(r'',text)

# text0 = "Why dont we make most grocery stores pickup only for the next few weeks? Would reduce crowding, panic buying, contamination, strain on workers. @Google could help build an app in a day. Open each store for 2 hrs for seniors who might be less tech savvy. #COVID2019 #Coronavirus"
# text = u'This is a smiley face \U0001f602'
# print(text) # with emoji
# print(clean_data(text))
# print(clean_data(text0))

def convert_data(file_name):
    data_struct = []
    data = pd.read_csv(file_name, encoding = 'latin-1')
    for index, row in data.iterrows():
        tweet = row["Tweet"]
        sentiment = row["Sentiment"]
        data_struct.append([tweet, sentiment])
    
    return data_struct


#print(convert_data("./Datasets/Corona_NLP_train.csv"))

def cleaned_csv(file_name, new_file_name):
    df = pd.read_csv(file_name, encoding = 'latin-1') ### Name of file that we're cleaning
    tweet = df["OriginalTweet"]
    sentiment = df["Sentiment"]
    data = {'Tweet': tweet, 'Sentiment': sentiment}

    df = pd.DataFrame(data, columns = ["Tweet", "Sentiment"])

    #for elem in df["Tweet"]:
    #    cleaned = clean_data(elem)
    #    df["Tweet"] = df['Tweet'].replace(elem,cleaned)

    #for sent in df["Sentiment"]:
    #    if(sent == 'Extremely Positive'):
    #        df["Sentiment"] = df['Sentiment'].replace(sent, 'Positive')
    #    if(sent == 'Extremely Negative'):
    #        df["Sentiment"] = df['Sentiment'].replace(sent, 'Negative')
    
    #Supposed to be faster version
    for i in df.index:
        cleaned = clean_data(df["Tweet"][i])
        df["Tweet"] = df['Tweet'].replace(df["Tweet"][i],cleaned)
        if(df["Sentiment"][i] == 'Extremely Positive'):
            df["Sentiment"] = df['Sentiment'].replace(df["Sentiment"][i], 'Positive')
        if(df["Sentiment"][i] == 'Extremely Negative'):
            df["Sentiment"] = df['Sentiment'].replace(df["Sentiment"][i], 'Negative')
        if(df["Sentiment"][i] != 'Positive' or df["Sentiment"][i] != 'Negative'):
            df["Sentiment"] = df['Sentiment'].replace(df["Sentiment"][i], 'Neutral')

    df.to_csv(new_file_name, index = False) ### Name of the file where the cleaned data is going

cleaned_csv("./Datasets/Corona_NLP_train.csv", "./Datasets/Corona_NLP_train_clean.csv")
cleaned_csv("./Datasets/Corona_NLP_test.csv", "./Datasets/Corona_NLP_test_clean.csv")


def get_tweet(file_name):
    return convert_data(file_name)[0]


def get_sentiment(file_name):
    return convert_data(file_name)[1]