"""
   Models and Algorithms in NLP Applications (LDA-H506)

   Starter code for Assignment 2: Perceptron

   Miikka Silfverberg
"""

from sys import argv, stderr, stdout
import os
import numpy as np
import nltk

from data import read_semeval_datasets, evaluate, write_semeval
from paths import data_dir, results_dir

from random import seed, shuffle
seed(0)

# Bias token added to every example. This is equivalent to having a
# separate bias weight.
BIAS="BIAS"

# You can train the model using three different parameter estimation
# algorithms.
MODES=["basic","averaged","mira"]

# Set this to 1 to classify the test set after you have tuned all
# hyper-parameters. Don't classify the test set before you have
# finished tuning all hyper-parameters.
CLASSIFY_TEST_SET=0

def extract_features(data):
    """
        This is a bag-of-words feature extractor for document
        classification. The features of a document are simply its
        tokens. A BIAS token is added to every example.

        This function modifies @a data. For every ex in data, it adds
        a binary np.array ex["FEATURES"].

        No need to edit this function.
    """
    all_tokens = sorted(list({wf for ex in data["training"] 
                              for wf in ex["BODY"]+[BIAS]}))
    encoder = {wf:i for i,wf in enumerate(all_tokens)}
    
    for data_set in data.values():
        for i,ex in enumerate(data_set):
            feature_ids = list({encoder[tok] for tok in ex["BODY"]+[BIAS] 
                                if tok in encoder})
            ex["FEATURES"] = np.zeros(len(all_tokens))
            ex["FEATURES"][feature_ids] = 1

def custom_extract_features(data):
    """ 
        Implement your own feature extraction function here.

        The function should modify data by adding a binary np.array
        ex["FEATURES"] to each ex in data.
    """
    # Replace this with your own code.
    for data_set in data.values():
        for ex in data_set:
            ex['BODY'] = nltk.word_tokenize(ex['BODY'])

    extract_features(data)

class Perceptron:
    """ 
        This is a simple perceptron classifier.
    
        It is initialized with a training set. You can train the
        classifier for n epochs by calling train(). You can classify
        a data set using classify().

        The fields of this classifier are
        
        W      - The collection of model parameters.
                 W[klass] is the parameter vector for klass.
        Ws     - The sum of all parameters since the beginnig of time.
        Y      - A list of sentiment classes.
        N      - The number of updates that have been performed 
                 during training.
        Lambda - The lambda hyper-parameter for MIRA.
    """
    def __init__(self,training_data):
        self.Y = list({ex["SENTIMENT"] for ex in training_data})        
        # Initialize all parameters to 0.
        self.W = {klass:np.zeros(training_data[0]["FEATURES"].shape)
                  for klass in self.Y}
        self.Ws = {klass:np.zeros(training_data[0]["FEATURES"].shape)
                  for klass in self.Y}
        # Start at 1 to avoid NumPy division by 0 warnings.
        self.N = 1
        # This is the lambda hyper-parameter for MIRA. You can tune it
        # using the development set.
        self.Lambda = 1

    def classify_ex(self,ex,mode,training=0):
        """
            This function classifies a single example. The
            implementation of classification will depend on the
            parameter estimation mode. For example, when mode is
            "averaged", you should use the cumulative parameters Ws
            instead of the current parameters W. 

            The parameter training indicates whether we are training
            or not. This is important for the averaged perceptron
            algorithm and MIRA. When we're training, we should use the
            Percptron.W parameters, whereas, when we're labeling the
            development or test set, we should use Perceptron.Ws during
            classification.

            Implement your own classification for different values of
            mode.
        """

        print(self.W, ex, mode)
        if mode == "basic":
            return "neutral"
        elif mode == "averaged":
            return "neutral"
        elif mode == 'mira':
            return "neutral"
        else:
            assert(0)

    def classify(self,data,mode):
        """
            This function classifies a data set. 

            No need to change this function.
        """
        return [self.classify_ex(ex,mode) for ex in data]

    def estimate_ex(self,ex,mode):
        """
            This function trains on a single example.

            You should edit it to implement parameter estimation for
            the different estimation modes.
        """
        gold_class = ex["SENTIMENT"]
        sys_class = self.classify_ex(ex,mode,training=1)

        if mode == "basic":
            pass
        elif mode == "averaged":
            pass
        elif mode == 'mira':
            # Compute and apply the MIRA update. You should call
            # get_eta() to compute the learning rate.
            pass
        else:
            assert(0)

    def get_eta(self,ex,sys_class,gold_class):
        """
            This function computes the learning rate eta for the MIRA
            estimation algorithm.

            Edit it to compute eta properly.
        """
        return 1

    def train(self,train_data,dev_data,mode,epochs):
        """
            This function trains the model. 

            No need to change this function.
        """
        for n in range(epochs):
            shuffle(train_data)
            stdout.write("Epoch %u : " % (n+1))
            for ex in train_data:
                self.N += 1
                self.estimate_ex(ex,mode)
            sys_classes = self.classify(dev_data,mode)
            acc, _ = evaluate(sys_classes, dev_data)
            print("Dev accuracy %.2f%%" % acc)
            
if __name__=="__main__":
#    if len(argv) != 3:
#        print("ERROR: Wrong number of command-line arguments.",file=stderr)
#        print("USAGE: %s data_directory output_directory" % argv[0],
#              file=stderr)
#        exit(1)
        
    # Read training, development and test sets and open the output for
    # test data.
    print("Reading data (this may take a while).")
    data = read_semeval_datasets(data_dir)

    output_file = open(os.path.join(results_dir,"test.output.txt"),
                       encoding="utf-8",
                       mode="w")

#    data_dir = argv[1]
#    data = read_semeval_datasets(data_dir)

#    output_dir = argv[2]
#    output_files = {mode:open("%s/test.output.%s.txt" % (output_dir, mode),
#                              encoding="utf-8",
#                              mode="w")
#                    for mode in MODES}

    print("Extracting features.")
    custom_extract_features(data)

    # Use the development set to tune the number of training epochs.
    epochs = {"basic":1, "averaged":1, "mira":1}

    for mode in MODES:
        print("Training %s model." % mode)
        model = Perceptron(data["training"])
        model.train(data["training"], data["development.gold"], mode, 
                    epochs[mode])

        if CLASSIFY_TEST_SET:
            print("Labeling test set.")
            test_output = model.classify(data["test.input"], mode)
            acc, fscores = evaluate(test_output, data["test.gold"])
            print("Final test accuracy: %.2f" % acc)
            print("Per class F1-fscore:")
            for c in fscores:
                print(" %s %.2f" % (c,fscores[c]))
            write_semeval(data["test.input"], test_output, output_files[mode])
        print()
