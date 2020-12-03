import matplotlib.pyplot as plt
import numpy as np

def graph():
    N = 4
    accuracy = (0.6451, 0.7915, 0.9134, 0.6793)
    precision = (0.6997, 0.7842, 0.9048, 0.6737)
    recall = (0.5278, 0.7498, 0.9122, 0.6374)

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