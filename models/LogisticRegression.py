import pandas as pd
from collections import defaultdict
from sklearn.feature_extraction import DictVectorizer
from sklearn.linear_model import LogisticRegressionCV
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score

def convert_data(file_name, column):
    data_struct = []
    data = pd.read_csv(file_name)
    for index, row in data.iterrows():
        data_struct.append(row[column])
    
    return data_struct

def print_confusion_matrix(file_nameA, file_nameB, column):
    choices_A = convert_data(file_nameA, column)
    choices_B = convert_data(file_nameB, column)
    print(confusion_matrix(choices_A, choices_B))

# from HW1
def tokenize_doc(doc):
    # From GEEKSFORGEEKS
    # removes punctuation from string
    punc = '''!()-[]{};:'"\,<>./?@#$%^&*_~'''
    for ele in doc:  
        if ele in punc:  
            doc = doc.replace(ele, "")

    # From HW1
    bow = defaultdict(float)
    tokens = doc.split()
    lowered_tokens = map(lambda t: t.lower(), tokens)
    for token in lowered_tokens:
        bow[token] += 1.0
    return dict(bow)

def LogisticRegressionCV_classifier(file_name, column):
    X = get_token_vectors(file_name)
    train_Y = convert_data(file_name, column)
    clf = LogisticRegressionCV(cv=5, random_state=0).fit(X, train_Y)
    y_pred = clf.predict(X)
    print("Metrics for LR classifier for " + column + ":")
    print("Accuracy Score: " + str(accuracy_score(train_Y, y_pred)))
    print("Precision Score " + str(calculate_precision(train_Y, y_pred)))
    print("Recall Score " + str(calculate_recall(train_Y, y_pred)))

def get_token_vectors(file_name):
    train_X = convert_data(file_name, 'txt')

    tokenized_sentences = []
    for sentence in train_X:
        token_array = tokenize_doc(sentence)
        tokenized_sentences.append(token_array)
    v = DictVectorizer(sparse=False)
    X = v.fit_transform(tokenized_sentences)
    return X

def ratio(file_name):
    opinion = convert_data(file_name, 'Opinion')
    proper_noun = convert_data(file_name, 'Proper Noun')
    count_bothY = 0
    count_bothN = 0
    count_proper_not_opinion = 0
    for i in range (0, len(opinion)):
        if (opinion[i] == proper_noun[i]):
            if(opinion[i] == 'Y'):
                count_bothY += 1
            else:
                count_bothN +=1
        else:
            if(proper_noun[i] == 'Y'):
                count_proper_not_opinion += 1
    print("Y agree= " + str(count_bothY))
    print("N agree= " + str(count_bothN))
    print("False Positive= " + str(count_proper_not_opinion))
    print("False Negative " + str(250 - count_bothY - count_bothN - count_proper_not_opinion))
    print("\n")


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

def Opinion_Proper_Noun_correlation(file_name):
    opinion_data = convert_data(file_name, "Opinion")
    PN_data = convert_data(file_name, "Proper Noun")
    print("Proper Noun vs Opinion statistics: ")
    print("Precision score : " + str(calculate_precision(opinion_data, PN_data)))
    print("Recall score: " + str(calculate_recall(opinion_data, PN_data)))


def top_ten(file_name, column):
    train_X = convert_data(file_name, 'txt')
    train_Y = convert_data(file_name, column)

    word_count_Y = defaultdict(float)
    word_count_N = defaultdict(float)
    index = 0
    for sentence in train_X:
        token_array = tokenize_doc(sentence)
        if(train_Y[index] == 'Y'):
            for word in token_array:
                word_count_Y[word] += token_array[word]
        else:
            for word in token_array:
                word_count_N[word] += token_array[word]
        index += 1
    sort_dict_Y = sorted(word_count_Y.items(), key=lambda x: x[1], reverse=True)
    sort_dict_N = sorted(word_count_N.items(), key=lambda x: x[1], reverse=True)
    i = 0
    print("Top ten words for POS label in column: " + column)
    for word in sort_dict_Y:
        if(i < 10):
            print(word[0], word[1])
            i += 1
        else: 
            break
    print(" ")
    print("Top ten words for NEG label in column: " + column)
    j = 0
    for word in sort_dict_N:
        if(j < 10):
            print(word[0], word[1])
            j += 1
        else: 
            break