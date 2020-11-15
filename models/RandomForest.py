import math
import numpy as np
from collections import defaultdict
from sklearn.feature_extraction import DictVectorizer
from sklearn.metrics import accuracy_score
from sklearn.ensemble import RandomForestClassifier

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
    tokenized_tweets = []
    for tweet in train_X:
        token_array = tokenize_doc(tweet)
        tokenized_tweets.append(token_array)
    v = DictVectorizer(sparse=False)
    X = v.fit_transform(tokenized_tweets)
    return X



def main():

    np.random.seed(0)

    train_struct = cd.convert_data('./Datasets/Corona_NLP_train_clean.csv')
    test_struct = cd.convert_data('./Datasets/Corona_NLP_test_clean.csv')



    train_X = get_token_vectors(np.array(train_struct)[:,0])
    train_y = np.array(train_struct)[:,1]
    #print(train_X.shape)
    #print(train_y.shape)

    test_X = get_token_vectors(np.array(test_struct)[:,0])
    test_y = np.array(test_struct)[:,1]
    #print(test_X.shape)
    #print(test_y.shape)




    RF = RandomForestClassifier(n_estimators=5, max_depth=3, random_state=0)
    RF.fit(train_X, train_y)
    y_pred = RF.predict(test_X)

    print(accuracy_score(y_pred, test_y))
 
    

if __name__ == '__main__':
    main()
