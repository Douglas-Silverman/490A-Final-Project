import pandas as pd
import numpy as np



import matplotlib.pyplot as plt
import math
import os
import time
import operator
from collections import defaultdict


from sklearn.naive_bayes import GaussianNB
import ConvertData as cd

# Global class labels.
POS_LABEL = 'Positive'
NEU_LABEL = 'Neutral'
NEG_LABEL = 'Negative'


###### DO NOT MODIFY THIS FUNCTION #####
def tokenize_doc(doc):
    """
    Tokenize a document and return its bag-of-words representation.
    doc - a string representing a document.
    returns a dictionary mapping each word to the number of times it appears in doc.
    """
    bow = defaultdict(float)
    tokens = doc.split()
    lowered_tokens = map(lambda t: t.lower(), tokens)
    for token in lowered_tokens:
        bow[token] += 1.0
    return dict(bow)
###### END FUNCTION #####



class NaiveBayes:
    """A Naive Bayes model for text classification."""

    def __init__(self, tokenizer):
        # Vocabulary is a set that stores every word seen in the training data
        self.vocab = set()
        self.tokenize_doc = tokenizer
        # class_total_doc_counts is a dictionary that maps a class (i.e., pos/neg) to
        # the number of documents in the trainning set of that class
        self.class_total_doc_counts = { POS_LABEL: 0.0,
                                        NEU_LABEL: 0.0,
                                        NEG_LABEL: 0.0 }

        # class_total_word_counts is a dictionary that maps a class (i.e., pos/neg) to
        # the number of words in the training set in documents of that class
        self.class_total_word_counts = { POS_LABEL: 0.0,
                                         NEU_LABEL: 0.0,
                                         NEG_LABEL: 0.0 }

        # class_word_counts is a dictionary of dictionaries. It maps a class (i.e.,
        # pos/neg) to a dictionary of word counts. For example:
        #    self.class_word_counts[POS_LABEL]['awesome']
        # stores the number of times the word 'awesome' appears in documents
        # of the positive class in the training documents.
        self.class_word_counts = { POS_LABEL: defaultdict(float),
                                   NEU_LABEL: defaultdict(float),
                                   NEG_LABEL: defaultdict(float) }

    def train_model(self):
        """
        This function processes the entire training set using the global PATH
        variable above.  It makes use of the tokenize_doc and update_model
        functions you will implement.
        """
        dataset = cd.convert_data("./Datasets/Corona_NLP_train_clean.csv")
        for row in dataset:
            content = row[0] # Tweet
            label = row[1] # Sentiment
            self.tokenize_and_update_model(content, label)
        self.report_statistics_after_training()

    def report_statistics_after_training(self):
        """
        Report a number of statistics after training.
        """

        print ("REPORTING CORPUS STATISTICS")
        print ("NUMBER OF DOCUMENTS IN POSITIVE CLASS:", self.class_total_doc_counts[POS_LABEL])
        print ("NUMBER OF DOCUMENTS IN NEUTRAL CLASS:", self.class_total_doc_counts[NEU_LABEL])
        print ("NUMBER OF DOCUMENTS IN NEGATIVE CLASS:", self.class_total_doc_counts[NEG_LABEL])
        print ("NUMBER OF TOKENS IN POSITIVE CLASS:", self.class_total_word_counts[POS_LABEL])
        print ("NUMBER OF TOKENS IN NEUTRAL CLASS:", self.class_total_word_counts[NEU_LABEL])
        print ("NUMBER OF TOKENS IN NEGATIVE CLASS:", self.class_total_word_counts[NEG_LABEL])
        print ("VOCABULARY SIZE: NUMBER OF UNIQUE WORDTYPES IN TRAINING CORPUS:", len(self.vocab))

    def update_model(self, bow, label):
        """
        IMPLEMENT ME!

        Update internal statistics given a document represented as a bag-of-words
        bow - a map from words to their counts
        label - the class of the document whose bag-of-words representation was input
        This function doesn't return anything but should update a number of internal
        statistics. Specifically, it updates:
          - the internal map the counts, per class, how many times each word was
            seen (self.class_word_counts)
          - the number of words seen for each label (self.class_total_word_counts)
          - the vocabulary seen so far (self.vocab)
          - the number of documents seen of each label (self.class_total_doc_counts)
        """
        for word in bow:
            # add word to vocab
            self.vocab.add(word)
            # increment total tokens
            self.class_total_word_counts[label] += 1
            # increment word count
            self.class_word_counts[label][word] += bow[word]
        
        # increment number of documents seen (only 1 for each bow)
        self.class_total_doc_counts[label] += 1
            
        # pass
        

    def tokenize_and_update_model(self, doc, label):
        """
        IMPLEMENT ME!

        Tokenizes a document doc and updates internal count statistics.
        doc - a string representing a document.
        label - the sentiment of the document (either postive or negative)
        
        Make sure when tokenizing to lower case all of the tokens!
        """
        bow = tokenize_doc(doc)
        self.update_model(bow, label)
        # pass

    def top_n(self, label, n):
        """
        Implement me!
        
        Returns the most frequent n tokens for documents with class 'label'.
        """
        sorted_list = [] # list of tuples
        for word in self.class_word_counts[label]:
            sorted_list.append((word, self.class_word_counts[label][word]))

        def sortSecond(val): 
            return val[1] 
        sorted_list.sort(key= sortSecond, reverse= True)
        
        n_list = []
        
        for i in range(0, n):
            n_list.append(sorted_list[i])   
        return n_list

        # pass

    def p_word_given_label(self, word, label):
        """
        Implement me!

        Returns the probability of word given label
        according to this NB model.
        """

        return self.class_word_counts[label][word] / self.class_total_word_counts[label]
        # pass

    def p_word_given_label_and_alpha(self, word, label, alpha):
        """
        Implement me!

        Returns the probability of word given label wrt psuedo counts.
        alpha - pseudocount parameter
        """
        return (self.class_word_counts[label][word] + alpha) / (self.class_total_word_counts[label] + (alpha * len(self.vocab)))
        # pass

    def log_likelihood(self, bow, label, alpha):
        """
        Implement me!

        Computes the log likelihood of a set of words give a label and pseudocount.
        bow - a bag of words (i.e., a tokenized document)
        label - either the positive or negative label
        alpha - float; pseudocount parameter
        """
        log_sum = 0.0
        for word in bow:
            log_sum += math.log(self.p_word_given_label_and_alpha(word, label, alpha), 2)
        return log_sum
        # pass

    def log_prior(self, label):
        """
        Implement me!

        Returns the log prior of a document having the class 'label'.
        """
        return self.class_total_doc_counts[label] / self.class_total_doc_counts[POS_LABEL] + self.class_total_doc_counts[NEG_LABEL] + self.class_total_doc_counts[NEU_LABEL]
        # pass

    def unnormalized_log_posterior(self, bow, label, alpha):
        """
        Implement me!

        Computes the unnormalized log posterior (of doc being of class 'label').
        bow - a bag of words (i.e., a tokenized document)
        """
        return self.log_prior(label) + self.log_likelihood(bow, label, alpha)
        # pass

    def classify(self, bow, alpha):
        """
        Implement me!

        Compares the unnormalized log posterior for doc for both the positive
        and negative classes and returns the either POS_LABEL, NEG_LABEL, or NEU_LABEL
        (depending on which resulted in the higher unnormalized log posterior)
        bow - a bag of words (i.e., a tokenized document)
        """

        POS_sum = self.unnormalized_log_posterior(bow, POS_LABEL, alpha)
        NEU_sum = self.unnormalized_log_posterior(bow, NEU_LABEL, alpha)
        NEG_sum = self.unnormalized_log_posterior(bow, NEG_LABEL, alpha)
        max_sum = POS_sum
        # if(max_sum < NEG_sum):
        #     return NEG_LABEL
        # return POS_LABEL
        return min(POS_sum, NEG_sum, NEU_sum)
        # pass


    def likelihood_ratio(self, word, alpha):
        """
        Implement me!

        Returns the ratio of P(word|pos) to P(word|neg).
        """

        return (self.p_word_given_label_and_alpha(word, POS_LABEL, alpha) / self.p_word_given_label_and_alpha(word, NEG_LABEL, alpha))
        # pass

    def evaluate_classifier_accuracy(self, alpha):
        """
        DO NOT MODIFY THIS FUNCTION

        alpha - pseudocount parameter.
        This function should go through the test data, classify each instance and
        compute the accuracy of the classifier (the fraction of classifications
        the classifier gets right.
        """
        correct = 0.0
        total = 0.0

        # pos_path = os.path.join(self.test_dir, POS_LABEL)
        # neg_path = os.path.join(self.test_dir, NEG_LABEL)
        # for (p, label) in [ (pos_path, POS_LABEL), (neg_path, NEG_LABEL) ]:
        #     for f in os.listdir(p):
        #         with open(os.path.join(p,f),'r', encoding='utf8') as doc:
        #             content = doc.read()
        #             bow = self.tokenize_doc(content)
        #             if self.classify(bow, alpha) == label:
        #                 correct += 1.0
        #             total += 1.0
        # return 100 * correct / total
        dataset = cd.convert_data("./Datasets/Corona_NLP_train_clean.csv")
        for row in dataset:
            content = row[0] # Tweet
            label = row[1] # Sentiment
            bow = self.tokenize_doc(content)
            if self.classify(bow, alpha) == label:
                correct += 1.0
            total += 1.0
        return 100 * correct / total
            


    # def find_wrong_review(self, alpha):
    #     for (p, label) in [ (pos_path, POS_LABEL), (neg_path, NEG_LABEL) ]:
    #         for f in os.listdir(p):
    #             with open(os.path.join(p,f),'r', encoding='utf8') as doc:
    #                 content = doc.read()
    #                 bow = self.tokenize_doc(content)
    #                 if self.classify(bow, alpha) != label:
    #                     return content, label


def main(): 
    train_set = './Datasets/Corona_NLP_train_clean.csv'
    nb = NaiveBayes(tokenizer=tokenize_doc)
    nb.train_model()

    print()
    print(nb.evaluate_classifier_accuracy(0.2))

if __name__ == '__main__':
    main()