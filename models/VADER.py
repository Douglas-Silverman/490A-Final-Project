import numpy as np

from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer

import ConvertData as cd

from sklearn.metrics import accuracy_score
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score

### used for debugging
# def sentiment_analyzer_scores(analyzer, sentence):
#     score = analyzer.polarity_scores(sentence)
#     print("{:-<40} {}".format(sentence, str(score)))


### predicts each tweets sentiment
### returns an array of predictions
def vader_predict(analyzer, tweets):
    predictions = []
    for tweet in tweets:
        VADER_dict = analyzer.polarity_scores(tweet)
        if(VADER_dict['compound'] >= 0.05):
            predictions.append('Positive')
        elif(VADER_dict['compound'] <= -0.05):
            predictions.append('Negative')
        else:
            predictions.append('Neutral')
    return predictions

def main():
    analyzer = SentimentIntensityAnalyzer() # creates analyzer
    test_struct = cd.convert_data('./Datasets/Corona_NLP_test_clean.csv') # obtain test data

    test_X = list(np.array(test_struct)[:,0]) 
    test_Y = np.array(test_struct)[:,1]

    y_pred = vader_predict(analyzer, test_X) # predicts tweets based on VADER

    accuracy = accuracy_score(test_Y, y_pred)
    precision = precision_score(test_Y, y_pred, average= 'macro')
    recall = recall_score(test_Y, y_pred, average= 'macro')

    print()
    print("VADER predictions:")
    print()
    print("\t accuracy: ", accuracy)
    print("\t precision: ", precision)
    print("\t recall: ", recall)

    return [accuracy, precision, recall]
    
 
    

if __name__ == '__main__':
    main()