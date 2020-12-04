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
    vectorizer = TfidfVectorizer(encoding= 'latin-1', stop_words= 'english')
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


def get_top_n(train_file_name, top_n):
    train_struct = cd.convert_data(train_file_name)

    train_array = np.array(train_struct)

    vectorizer = get_token_vectors(train_array[:,0])
    train_X = vectorizer[0] #Xtr
    v = vectorizer[1]

    display_scores(v, train_X, top_n)

def display_scores(vectorizer, tfidf_result, top_n):
    # http://stackoverflow.com/questions/16078015/
    scores = zip(vectorizer.get_feature_names(),
                 np.asarray(tfidf_result.sum(axis=0)).ravel())
    sorted_scores = sorted(scores, key=lambda x: x[1], reverse=True)
    print("Top " + str(top_n)+ " Features: \n")
    for i in range(0,top_n):
        item = sorted_scores[i]
        print(str(item[0]) +": " + str(item[1]))

def main():
    # LogisticRegression_classifier('./Datasets/Corona_NLP_train_clean.csv', './Datasets/Corona_NLP_test_clean.csv')
    get_top_n('./Datasets/Corona_NLP_train_clean.csv', 10)

if __name__ == '__main__':
    main()
