import matplotlib.pyplot as plt
import numpy as np

import NaiveBayes as NB
import LogisticRegression as LR
import VADER as VD
import RandomForest as RF


### hardcoded graph data from individual outputs from each classifier
def graph():
    NB_output = NB.NaiveBayes_classifier('./Datasets/Corona_NLP_train_clean.csv', './Datasets/Corona_NLP_test_clean.csv')
    LR_output = LR.LogisticRegression_classifier('./Datasets/Corona_NLP_train_clean.csv', './Datasets/Corona_NLP_test_clean.csv')
    VD_output = VD.main()
    RF_output = RF.RandomForest_classifier('./Datasets/Corona_NLP_train_clean.csv', './Datasets/Corona_NLP_test_clean.csv')
    
    
    N = 4

    accuracy = (NB_output[0], LR_output[0], VD_output[0], RF_output[0])
    precision = (NB_output[1], LR_output[1], VD_output[1], RF_output[1])
    recall = (NB_output[2], LR_output[2], VD_output[2], RF_output[2])


    # accuracy = (0.6451, 0.7915, 0.9134, 0.6793)
    # precision = (0.6997, 0.7842, 0.9048, 0.6737)
    # recall = (0.5278, 0.7498, 0.9122, 0.6374)

    ind = np.arange(N) 
    width = 0.2       
    plt.bar(ind, accuracy, width, label='F1')
    plt.bar(ind + width, precision, width,
        label='Precision')
    plt.bar(ind + width + width, recall, width,
        label='Recall')

    plt.ylabel('Scores')
    plt.title('Performance of Classifiers')

    plt.xticks(ind + width / 2, ('Naive Bayes', 'Logistic Regression', 'VADER', 'Random Forests'))
    plt.legend(loc='best')
    plt.show()


def main():
    graph()

if __name__ == '__main__':
    main()