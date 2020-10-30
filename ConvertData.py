import pandas as pd



def convert_data(file_name):
    data_struct = []
    data = pd.read_csv(file_name)
    for index, row in data.iterrows():
        tweet = row["OriginalTweet"]
        sentiment = row["Sentiment"]
        data_struct.append([tweet, sentiment])
    
    return data_struct


print(pd.read_csv("./Datasets/Corona_NLP_train.csv", encoding= 'utf-8'))