import pandas as pd
import numpy as np
from collections import defaultdict
from sklearn.feature_extraction import DictVectorizer
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB


import ConvertData as cd

def get_token_vectors(tweets):
    train_X = tweets # get the tweets
    # tokenized_tweets = []
    # for tweet in train_X:
    #     token_array = tokenize_doc(tweet)
    #     tokenized_tweets.append(token_array)
    # v = DictVectorizer(sparse=False)
    # X = v.fit_transform(tokenized_tweets)
    vectorizer = TfidfVectorizer(encoding= 'latin-1')
    X = vectorizer.fit_transform(tweets)

    # print(X)
    # print(v.inverse_transform(X))
    return X, vectorizer

def get_token_vectors_test(tweets, v):
    # test_X = tweets # get the tweets
    # tokenized_tweets = []
    # for tweet in test_X:
    #     token_array = tokenize_doc(tweet)
    #     tokenized_tweets.append(token_array)
    X = v.transform(tweets)
    return X


    
def NaiveBayes_classifier(train_file_name, test_file_name):
    train_struct = cd.convert_data(train_file_name)
    test_struct = cd.convert_data(test_file_name)

    train_array = np.array(train_struct)
    test_array = np.array(test_struct)


    vectorizer = get_token_vectors(train_array[:,0])
    train_X = vectorizer[0]
    v = vectorizer[1]
    train_Y = train_array[:,1]

    
    print(train_X.shape)

    test_X = get_token_vectors_test(test_array[:,0], v)
    print(test_X.shape)

    test_Y = test_array[:,1]


    clf = MultinomialNB(alpha = 0.2)
    print("start training")
    clf.fit(train_X, train_Y)
    print("training done")
    y_pred = clf.predict(test_X)

    print("\t accuracy: ", accuracy_score(test_Y, y_pred))


def main():
    NaiveBayes_classifier('./Datasets/Corona_NLP_train_clean.csv', './Datasets/Corona_NLP_test_clean.csv')

if __name__ == '__main__':
    main()