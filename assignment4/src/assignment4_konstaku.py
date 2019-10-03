"""
   Models and Algorithms in NLP Applications (LDA-H506)

   Starter code for Assignment 3: Logistic Regression

   Miikka Silfverberg
"""

from sys import argv, stderr, stdout
from random import seed, shuffle
import os

import numpy as np
from scipy.misc import logsumexp
import scipy
import nltk

seed(0)

from data import evaluate, get_class, read_20newsgroup_datasets
from paths import data_dir, results_dir

# Bias token added to every example. This is equivalent to having a
# separate bias weight.
BIAS="BIAS"
FEATURES='FEATURES'
CLASS='CLASS'

# When reduction in loss is less than DELTA_THRESHOLD, interrupt
# training.
DELTA_THRESHOLD = 1

def extract_features(data):
    """
        This is a bag-of-words feature extractor for document
        classification. The features of a document are simply its
        tokens. A BIAS token is added to every example.

        This function modifies @a data. For every ex in data, it adds
        a binary np.array ex["FEATURES"].

        No need to edit this function.
    """
    # Replace this with your own code.
    all_tokens = sorted(list({wf for ex in data["train"]
                              for wf in ex["BODY"]+[BIAS]}))
    encoder = {wf:i for i,wf in enumerate(all_tokens)}
    
    for data_set in data.values():
        for i,ex in enumerate(data_set):            
            ex["FEATURES"] = np.zeros(len(all_tokens))
            for fi in list(encoder[tok] for tok in ex["BODY"]+[BIAS] 
                           if tok in encoder):
                ex["FEATURES"][fi] = 1

"""
    This function receives as input an array:

      [log p(positive,x), log p(neutral,x), log p(negative,x)]

    It returns an array:

      [p("positive"|x), p("neutral"|x), p("negative"|x)]

    It is your task to implement normalization properly.
"""
def normalize_ll(log_likelihoods):
    total = scipy.special.logsumexp(log_likelihoods)
    return [np.exp(p - total) for p in log_likelihoods]

class NaiveBayes:
    """ 
        This is a simple semi-supervised Naive Bayes model.
    
        It is initialized with a training set. You can train the
        classifier using a labeled and unlabeled traing set using the
        function train(). You can classify a data set using
        classify().

        The fields of this classifier are
        
        Y              - A list of sentiment classes.
        alpha          - Smoothing term for Laplace smoothing.
        Lambda         - Learning rate.
        N              - Class counts.
        joint_counts   - Count(w,y) for word w and class y.
        feature_counts - sum_w' Count(w',y) for class y.
    """
    def __init__(self,training_data,Lambda=1):
        self.Y = list({ex["CLASS"] for ex in training_data})        

        # Alpha for Laplace smoothing.
        self.alpha = 1

        # Learning rate for EM.
        self.Lambda = Lambda

        self.N = {klass:0 for klass in self.Y}
        self.joint_counts = {klass:np.zeros(training_data[0]["FEATURES"].shape)
                      for klass in self.Y}
        self.feature_counts = {klass:0 for klass in self.Y}

    def classify_ex(self,ex):
        """
            Classify one example using the model.

            No need to change this function.
        """
        log_likelihoods = []
        for klass in self.Y:
            class_prior = np.log(self.N[klass] / sum(self.N.values()))
            smooth_w_y = self.joint_counts[klass] + self.alpha
            smooth_any_y = (self.feature_counts[klass] + 
                            ex["FEATURES"].shape[0] * self.alpha)
            likelihood = ((np.log(smooth_w_y) - np.log(smooth_any_y)) @ 
                          ex["FEATURES"])
            log_likelihoods.append((class_prior + likelihood) * 
                                   1 / np.sum(ex["FEATURES"]))

        return log_likelihoods

    def get_class(self,likelihoods):
        """
            Given a distribution over labels, return the most
            probable label.

            No need to change this function.
        """
        return self.Y[np.argmax(likelihoods)]

    def classify(self,data):
        """
            This function classifies a data set. 

            No need to change this function.
        """
        return [self.get_class(self.classify_ex(ex)) for ex in data]

    def update_ex(self,ex):
        """
            This function performs Naive Bayes updates for one labeled
            example.

            No need to change this function.
        """
        self.N[ex["CLASS"]] += 1
        self.joint_counts[ex["CLASS"]] += ex["FEATURES"]
        self.feature_counts[ex["CLASS"]] += np.sum(ex["FEATURES"])

    def soft_update_ex(self,ex):
        """
            This function performs Naive Bayes updates for one unlabeled
            example.

            It is your task to perform updates properly.
        """
        feats = ex[FEATURES]
        distr = normalize_ll(self.classify_ex(ex))
        for c in self.Y:
            self.N[c] += 1 * distr[int(c) - 1] * self.Lambda
            self.joint_counts[c] += feats * distr[int(c) - 1] * self.Lambda
            self.feature_counts[c] += np.sum(feats * distr[int(c) - 1] * self.Lambda)
        return distr

    def get_loss(self,labeled_train_data,unlabeled_train_data):
        """
            This function computes the loss over the labeled and
            unlabeled training examples.

            It is your task to compute the loss properly.
        """
        loss = np.float64(0)

        for ex in labeled_train_data:
            log_p = self.classify_ex(ex)[int(ex[CLASS]) - 1]
            loss -= log_p

        for ex in unlabeled_train_data:   
            assert(type(loss) == np.float64)
            loss -= logsumexp(self.classify_ex(ex))

        return loss

    def train(self,labeled_train_data,unlabeled_train_data,semisupervised=1):
        """
            This function trains the model. 

            No need to change this function.
        """
        for ex in labeled_train_data:
            self.update_ex(ex)
            
        if not semisupervised:
            return

        oldloss = self.get_loss(labeled_train_data,unlabeled_train_data)
        while 1:
            for ex in labeled_train_data:
                self.update_ex(ex)

            for ex in unlabeled_train_data:
                self.soft_update_ex(ex)

            loss = self.get_loss(labeled_train_data,unlabeled_train_data)
            if oldloss - loss < DELTA_THRESHOLD:
                break

            print("   LOSS: %.2f (DELTA: %.2f)" % (loss,oldloss-loss))
            oldloss = loss

        
if __name__=="__main__":
    # Read training and test sets.
    print("Reading data (this may take a while).")
    data = read_20newsgroup_datasets(data_dir)
    shuffle(data["train"])
    print("Extracting features.")
    extract_features(data)

    # You can explore the effect of lambda using the test set.
    lambdas = {10:0.1, 50:0.0001, 100:0.0001}

    for labeled_size in [10,50,100]:
        print("Experiment with %u labeled examples:" % labeled_size)
        labeled_train_data = data['train'][:labeled_size]
        unlabeled_train_data = data['train'][labeled_size:]

        print("  Training fully supervised model.")
        model = NaiveBayes(data['train'])
        model.train(labeled_train_data, unlabeled_train_data, semisupervised=0)

        test_output = model.classify(data["test"])
        acc, fscores = evaluate(test_output, data["test"])
        print("  Test accuracy: %.2f" % acc)
        print()

        print(" Training semi-supervised model.")
        model = NaiveBayes(data['train'],lambdas[labeled_size])
        model.train(labeled_train_data, unlabeled_train_data, semisupervised=1)

        test_output = model.classify(data["test"])
        acc, fscores = evaluate(test_output, data["test"])
        print("  Test accuracy: %.2f" % acc)
        print()
