import pandas as pd
import numpy as np
from collections import defaultdict
from sklearn.feature_extraction import DictVectorizer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression

from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score


import ConvertData as cd

# from HW1
def tokenize_doc(doc):
    # From HW1
    bow = defaultdict(float)
    tokens = doc.split()
    lowered_tokens = map(lambda t: t.lower(), tokens)
    for token in lowered_tokens:
        bow[token] += 1.0
    return dict(bow)

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


    
def LogisticRegression_classifier(train_file_name, test_file_name):
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


    clf = LogisticRegression(solver= 'lbfgs', multi_class= 'multinomial', max_iter= 1000)
    print("start training")
    clf.fit(train_X, train_Y)
    print("training done")
    y_pred = clf.predict(test_X)

    print("accuracy: ", accuracy_score(test_Y, y_pred))
    print("precision: ", precision_score(test_Y, y_pred, average= 'macro'))
    print("recall: ", recall_score(test_Y, y_pred, average= 'macro'))




    # print(clf.score(test_X, test_Y))

    """
    print("Metrics for LR classifier for Sentiment: ")
    print("Accuracy Score: " + str(accuracy_score(train_Y, y_pred)))
    print("Precision Score " + str(calculate_precision(train_Y, y_pred)))
    print("Recall Score " + str(calculate_recall(train_Y, y_pred)))
    """


def calculate_precision(y_true, y_pred):
    Tp = 0.01
    Fp = 0.01
    for i in range (0, len(y_true)):
        if(y_pred[i] == 'Y'):
            if(y_true[i] == 'Y'):
                Tp += 1
            else:
                Fp += 1
    return Tp / (Tp + Fp)

def calculate_recall(y_true, y_pred):
    Tp = 0.01
    Fn = 0.01
    for i in range (0, len(y_true)):
        if(y_true[i] == 'Y'):
            if(y_pred[i] == 'Y'):
                Tp += 1
            else:
                Fn += 1
    return Tp / (Tp + Fn)


def main(): 
    LogisticRegression_classifier('./Datasets/Corona_NLP_train_clean.csv', './Datasets/Corona_NLP_test_clean.csv')

if __name__ == '__main__':
    main()