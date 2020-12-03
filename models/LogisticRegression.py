import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression

from sklearn.metrics import accuracy_score
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score


import ConvertData as cd

def get_token_vectors(tweets):
    train_X = tweets # get the tweets
    vectorizer = TfidfVectorizer(encoding= 'latin-1')
    X = vectorizer.fit_transform(tweets)
    return X, vectorizer

def get_token_vectors_test(tweets, v):
    X = v.transform(tweets)
    return X


    
def LogisticRegression_classifier(train_file_name, test_file_name):
    train_struct = cd.convert_data(train_file_name)
    test_struct = cd.convert_data(test_file_name)

    train_array = np.array(train_struct)
    test_array = np.array(test_struct)


    vectorizer = get_token_vectors(train_array[:,0])
    train_X = vectorizer[0] #Xtr
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

    print()
    print("Logistic Regression predictions:")
    print()
    print("\t accuracy: ", accuracy_score(test_Y, y_pred))
    print("\t precision: ", precision_score(test_Y, y_pred, average= 'macro'))
    print("\t recall: ", recall_score(test_Y, y_pred, average= 'macro'))

def main():
    LogisticRegression_classifier('./Datasets/Corona_NLP_train_clean.csv', './Datasets/Corona_NLP_test_clean.csv')

if __name__ == '__main__':
    main()