"""
   Models and Algorithms in NLP Applications (LDA-H506)

   Starter code for Assignment 3: Logistic Regression

   Miikka Silfverberg
"""

from sys import argv, stderr, stdout
import os
import numpy as np
import nltk
from random import seed, shuffle
seed(0)
from copy import deepcopy

from data import read_semeval_datasets, evaluate, write_semeval, get_class
from paths import data_dir, results_dir
from nltk.corpus import sentiwordnet as swn

# Bias token added to every example. This is equivalent to having a
# separate bias weight.
BIAS="BIAS"
FEATURES = "FEATURES"
# You can train the model using three different parameter estimation
# algorithms.
MODES=["basic","averaged","mira"]

# Set this to 1 to classify the test set after you have tuned all
# hyper-parameters. Don't classify the test set before you have
# finished tuning all hyper-parameters.
CLASSIFY_TEST_SET=1

def extract_features(data):
    """
        This is a bag-of-words feature extractor for document
        classification. The features of a document are simply its
        tokens. A BIAS token is added to every example.

        This function modifies @a data. For every ex in data, it adds
        a binary np.array ex["FEATURES"].

        No need to edit this function.
    """
    # Make encoder available globally (this is needed for analyzing
    # feature weights).
    global encoder
    tokenizer = nltk.TweetTokenizer(preserve_case=False, reduce_len=True)

    get_worst = lambda x: max([e.neg_score() for e in x])
    get_best = lambda x: max([e.pos_score() for e in x])

    # Replace this with your own code.
    for data_set in data.values():
        for ex in data_set:
            ex['BODY'] = [(word) for word in tokenizer.tokenize(ex['BODY']) if len(list(swn.senti_synsets(word))) == 0 or (get_best(list(swn.senti_synsets(word))) > 0.1 or get_worst(list(swn.senti_synsets(word))) > 0.1)]

    all_tokens = sorted(list({wf for ex in data["training"] 
                              for wf in ex["BODY"]+[BIAS]}))
    encoder = {wf:i for i,wf in enumerate(all_tokens)}
    
    for data_set in data.values():
        for i,ex in enumerate(data_set):
            feature_ids = list({encoder[tok] for tok in ex["BODY"]+[BIAS] 
                                if tok in encoder})
            ex["FEATURES"] = np.zeros(len(all_tokens))
            ex["FEATURES"][feature_ids] = 1

def softmax(scores):
    """
        This function implements softmax for the real numbers in the
        np.array scores. Provide a proper implementation for the
        function.
    """
    tot = np.sum([np.exp(v) for k, v in scores.items()])

    for k, v in scores.items():
        scores[k] = np.exp(v) / tot
    
    return scores

class LogisticRegression:
    """ 
        This is a simple logistic regression model.
    
        It is initialized with a training set. You can train the
        classifier for n epochs by calling train(). You can classify
        a data set using classify().

        The fields of this classifier are
        
        W      - The collection of model parameters.
                 W[klass] is the parameter vector for klass.
        Y      - A list of sentiment classes.
        N      - The number of updates that have been performed 
                 during training.
        Lambda - The learning rate.
    """
    def __init__(self,training_data):
        self.Y = list({ex["SENTIMENT"] for ex in training_data})        
        # Initialize all parameters to 0.
        self.W = {klass:np.zeros(training_data[0]["FEATURES"].shape)
                  for klass in self.Y}
        self.N = 0
        # This is the lambda learning rate. You can tune it
        # using the development set.
        self.Lambda = 1

    def classify_ex(self,ex):
        """
            This function classifies a single example. 

            It returns a dictionary where the keys are classes
            (positive, neutral, negative) and the values are
            probabilities, for example 
                  p("positive"|ex["FEATURES"];self.W).

            Implement your own classification. You will need the
            weights in self.W and the featue vector
            ex["FEATURES"]. You should also make use of the function
            softmax().
        """
        return softmax({klass: np.dot(self.W[klass], ex[FEATURES]) for klass in self.Y})

    def classify(self,data):
        """
            This function classifies a data set. 

            No need to change this function.
        """
        return [get_class(self.classify_ex(ex)) for ex in data]

    def estimate_ex(self,ex):
        """
            This function trains on a single example.

            You should edit it to implement parameter estimation
            properly. You will need to call self.classify_ex() and you
            need to use the feature vector ex["FEATURES"].

            You will also need the learning rate self.Lambda and the
            parameters self.W.
        """
        gold_class = ex["SENTIMENT"]
        sys_class_distribution = self.classify_ex(ex)
        
        for c, p in sys_class_distribution.items():
            if c == gold_class:
                self.W[c] += self.Lambda * (1 - p) * ex[FEATURES]
            else:
                self.W[c] -= self.Lambda * p * ex[FEATURES]

    def train(self,train_data,dev_data,epochs):
        """
            This function trains the model. 

            No need to change this function.
        """
        acc = 0
        best_weights = {}
        best_acc = 0
        for n in range(epochs):
            shuffle(train_data)
            stdout.write("Epoch %u : " % (n+1))
            for ex in train_data:
                self.N += 1
                self.estimate_ex(ex)
            sys_classes = self.classify(dev_data)
            acc, _ = evaluate(sys_classes, dev_data)
            
            sys_classes_1 = self.classify(train_data)
            t_acc, _ = evaluate(sys_classes_1, train_data)

            print("Train accuracy %.2f%%, Dev accuracy %.2f%%" % (t_acc, acc))

            if acc > best_acc:
                best_weights = self.W
                best_acc = acc
                print('New best val accuracy model for these parameters')
            if t_acc > 99:
                print('Training accuracy preeetty good for now pls stop mister')
                self.W = best_weights
                return best_acc

        self.W = best_weights
        return best_acc
            
if __name__=="__main__":
    # Read training, development and test sets and open the output for
    # test data.
    print("Reading data (this may take a while).")
    data = read_semeval_datasets(data_dir)

    output_file = open(os.path.join(results_dir,"test.output.txt"),
                       encoding="utf-8",
                       mode="w")

    print("Extracting features.")
    extract_features(data)

    # Use the development set to tune the number of training epochs.
    epochs = 20

    print("Training model.")
    best_model = (0, 0)
    for l in [0.1, 0.01, 0.001, 1]:
        print(f'Trying with learning rate {l}')
        model = LogisticRegression(data["training"])
        model.Lambda = l
        val_acc = model.train(data["training"], data["development.gold"], epochs)
        if val_acc > best_model[1]:
            best_model = (deepcopy(model), val_acc)
            print(f'Found new best model with lambda of {l} and val accuracy of {val_acc} ')

    if CLASSIFY_TEST_SET:
        print("Labeling test set with the best val accuracy model.")
        test_output = best_model[0].classify(data["test.input"])
        acc, fscores = evaluate(test_output, data["test.gold"])
        print("Final test accuracy: %.2f" % acc)
        print("Per class F1-fscore:")
        for c in fscores:
            print(" %s %.2f" % (c,fscores[c]))
        write_semeval(data["test.input"], test_output, output_file)
    print(f'Best val accuracy was {best_model[1]}')

    """
        Write your code for analyzing model weights here.

        You can use the Python dict encoder which will be defined.
    """

    model = best_model[0]
    for c in model.Y:
        for w in ['happy', 'great', 'awful', 'sucks']:
            weight = model.W[c][encoder[w]]
            corr = 'POSITIVE correlation' if weight >= 0 else 'NEGATIVE correlation'
            print(f'Class: {c}  -- Word: {w} -- Value: {weight} -- {corr}')

    for c in model.Y:
        biggest = np.argmax(model.W[c])
        smallest = np.argmin(model.W[c])
        print(f'Most impactful words for class {c}')
        for word, key in encoder.items():
            if key == biggest:
                print(f'Largest weight was for {word}, {model.W[c][biggest]}')
            if key == smallest:
                print(f'Smallest weight was for {word}, {model.W[c][smallest]}')
    
