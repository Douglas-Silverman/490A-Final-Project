import pandas as pd
from collections import defaultdict
from sklearn.feature_extraction import DictVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score


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

def LogisticRegressionCV_classifier(file_name):
    X = get_token_vectors(file_name)
    train_Y = cd.convert_data(file_name)[1]
    clf = LogisticRegression(multi_class= 'multinomial').fit(X, train_Y)
    y_pred = clf.predict(X)

    print("accuracy: ", accuracy_score(train_Y, y_pred))

    """
    print("Metrics for LR classifier for Sentiment: ")
    print("Accuracy Score: " + str(accuracy_score(train_Y, y_pred)))
    print("Precision Score " + str(calculate_precision(train_Y, y_pred)))
    print("Recall Score " + str(calculate_recall(train_Y, y_pred)))
    """

def get_token_vectors(file_name):
    train_X = cd.convert_data(file_name)[0] # get the tweets

    tokenized_tweets = []
    for tweet in train_X:
        token_array = tokenize_doc(tweet)
        tokenized_tweets.append(token_array)
    v = DictVectorizer(sparse=False)
    X = v.fit_transform(tokenized_tweets)
    return X


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
    LogisticRegressionCV_classifier('./Datasets/Corona_NLP_test.csv')

if __name__ == '__main__':
    main()