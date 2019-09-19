"""
   Models and Algorithms in NLP Applications (LDA-H506)

   Starter code for Assignment 2: Perceptron

   Miikka Silfverberg
"""

from sys import argv, stderr, stdout
import os
import operator
import numpy as np
import nltk
from collections import defaultdict
from data import read_semeval_datasets, evaluate, write_semeval
from paths import data_dir, results_dir
from nltk.stem import PorterStemmer
from nltk.corpus import sentiwordnet as swn

from nltk.corpus import stopwords
from random import seed, shuffle
seed(0)

# Bias token added to every example. This is equivalent to having a
# separate bias weight.
BIAS="BIAS"

# You can train the model using three different parameter estimation
# algorithms.
MODES=["basic","averaged","mira"]

FEATURES='FEATURES'

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
    
    def clean(word):
        return ''.join(filter(lambda x: x.isalnum(), word)) != ''

    counts = defaultdict(int)
    # all_tokens = (list({(wf for ex in data["training"] 
    #                           for wf in ex["BODY"]+[BIAS]}))

    # Add all uni, bi, tri and quadgrams
    for ex in data['training']:
        sent = list(filter(clean, (ex["BODY"] + [BIAS])))
        sent = ex["BODY"] + [BIAS]
        
        for ind, wf in enumerate(sent):
            counts[wf] += 1
        
        # for ind, wf in enumerate(sent[:-1]):
        #     counts[wf, sent[ind + 1]] += 1

        # for ind, wf in enumerate(sent[:-2]):
        #     counts[wf, sent[ind + 1], sent[ind + 2]] += 1

        for ind, wf in enumerate(sent[:-3]):
            counts[wf, sent[ind + 1], sent[ind + 2], sent[ind + 3]] += 1

    all_tokens = []
    feats= [0,0,0,0]

    # Add some special logic for what n grams are added to vocab
    for gram, count in counts.items():
        if type(gram) != tuple and count > 0:
            all_tokens.append(gram)
            feats[0] += 1

        if type(gram) == tuple and len(gram) == 2 and count > 4:
            all_tokens.append(gram)
            feats[1] += 1
        
        if type(gram) == tuple and len(gram) > 2 and count > 1:
            all_tokens.append(gram)
            feats[len(gram) - 1] += 1

    print(len(all_tokens), 'features in the vector')
    for ind, feat in enumerate(feats):
        print(f'Number of {ind + 1} - grams is {feat} ')

    encoder = {wf:i for i,wf in enumerate(all_tokens)}

    # add all n-grams to feature vector
    for data_set in data.values():
        for i,ex in enumerate(data_set):

            # debug print examples
            # if np.random.random() < 0.001:
            #     print(ex)

            ex["FEATURES"] = np.zeros(len(all_tokens))

            feature_ids = list({encoder[(tok, ((ex["BODY"]+[BIAS]))[ind +1], ((ex["BODY"]+[BIAS]))[ind +2], ((ex["BODY"]+[BIAS]))[ind +3])] for ind, tok in enumerate((ex["BODY"]+[BIAS])[:-3])
                                if (tok, ((ex["BODY"]+[BIAS]))[ind +1], ((ex["BODY"]+[BIAS]))[ind +2], ((ex["BODY"]+[BIAS]))[ind +3]) in encoder})
            ex["FEATURES"][feature_ids] = 1

            feature_ids = list({encoder[(tok, ((ex["BODY"]+[BIAS]))[ind +1], ((ex["BODY"]+[BIAS]))[ind +2])] for ind, tok in enumerate((ex["BODY"]+[BIAS])[:-2])
                                if (tok, ((ex["BODY"]+[BIAS]))[ind +1], ((ex["BODY"]+[BIAS]))[ind +2]) in encoder})
            ex["FEATURES"][feature_ids] = 1
            
            
            feature_ids = list({encoder[(tok, ((ex["BODY"]+[BIAS]))[ind +1])] for ind, tok in enumerate((ex["BODY"]+[BIAS])[:-1])
                                if (tok, ((ex["BODY"]+[BIAS]))[ind +1]) in encoder})

            ex["FEATURES"][feature_ids] = 1
            feature_ids = list({encoder[tok] for ind, tok in enumerate((ex["BODY"]+[BIAS]))
                                if tok in encoder})
            ex["FEATURES"][feature_ids] = 1

def custom_extract_features(data):
    """ 
        Implement your own feature extraction function here.

        The function should modify data by adding a binary np.array
        ex["FEATURES"] to each ex in data.
    """
    # Replace this with your own code.
    tokenizer = nltk.TweetTokenizer(preserve_case=False, reduce_len=True)
    stemmer = PorterStemmer()
    sw = {w: True for w in set(stopwords.words('english'))}

    get_worst = lambda x: max([e.neg_score() for e in x])
    get_best = lambda x: max([e.pos_score() for e in x])

    for data_set in data.values():
        for ex in data_set:
            # Words that are clearly loaded one way
            # ex['BODY'] = [(word) for word in tokenizer.tokenize(ex['BODY']) if len(list(swn.senti_synsets(word))) == 0 or np.abs(get_best(list(swn.senti_synsets(word))) - get_worst(list(swn.senti_synsets(word)))) > 0.25]
            
            # Preserve words not in the sentinet i.e. likely smileys, abbreviations etc and only keep the words that have at least some sentiment score
            ex['BODY'] = [(word) for word in tokenizer.tokenize(ex['BODY']) if len(list(swn.senti_synsets(word))) == 0 or (get_best(list(swn.senti_synsets(word))) > 0.1 or get_worst(list(swn.senti_synsets(word))) > 0.1)]

            # Filter stopwords
            # ex['BODY'] = [(word) for word in (tokenizer.tokenize(ex['BODY'])) if word not in sw]
            # ex['BODY'] = [(word) for word in (tokenizer.tokenize(ex['BODY']))]

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
        self.Lambda = 1e-1

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

        assert(len(self.W[self.Y[0]]) == len(ex[FEATURES]))

        if mode == "basic":
            return self.Y[np.argmax([np.dot(self.W[cls], ex[FEATURES]) for cls in self.Y])]
        elif mode == "averaged":
            if training:
                return self.Y[np.argmax([np.dot(self.W[cls], ex[FEATURES]) for cls in self.Y])]
            return self.Y[np.argmax([np.dot(self.Ws[cls] / self.N, ex[FEATURES]) for cls in self.Y])]
        elif mode == 'mira':
            if training:
                return self.Y[np.argmax([np.dot(self.W[cls], ex[FEATURES]) for cls in self.Y])]
            return self.Y[np.argmax([np.dot(self.Ws[cls] / self.N, ex[FEATURES]) for cls in self.Y])]
        else:
            assert(0)

    def classify(self,data,mode):
        """
            This function classifies a data set. 

            No need to change this function.
        """
        return [self.classify_ex(ex,mode) for ex in data]

    def handle_error(self, exp_class, pred_class, features):
        self.W[exp_class] += features
        self.W[pred_class] -= features

    def estimate_ex(self,ex,mode):
        """
            This function trains on a single example.

            You should edit it to implement parameter estimation for
            the different estimation modes.
        """
        gold_class = ex["SENTIMENT"]
        sys_class = self.classify_ex(ex,mode,training=1)
        correc_pred = 0 if gold_class == sys_class else 1
        feats = ex[FEATURES]

        if mode == "basic":
            if sys_class != gold_class:
                self.handle_error(gold_class, sys_class, feats)
        elif mode == "averaged":
            self.N += 1
            if sys_class != gold_class:
                self.handle_error(gold_class, sys_class, feats)
            for cls in self.Y:
                self.Ws[cls] += self.W[cls]
        elif mode == 'mira':
            # Compute and apply the MIRA update. You should call
            # get_eta() to compute the learning rate.
            tau = self.get_eta(feats, sys_class, gold_class)
            self.W[gold_class] += tau * (feats)
            self.W[sys_class] += -tau * (feats)
                
            self.N += 1
            for cls in self.Y:
                self.Ws[cls] += self.W[cls]
        else:
            assert(0)

    def get_eta(self,ex,sys_class,gold_class):
        """
            This function computes the learning rate eta for the MIRA
            estimation algorithm.

            Edit it to compute eta properly.
        """
        corr = -np.dot(self.W[gold_class], ex)
        pred = np.dot(self.W[sys_class], ex)
        loss = max(0, corr + pred) + int(gold_class != sys_class)
        b = (np.linalg.norm(ex) * 2)

        return min(1/self.Lambda, loss / b)

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
            
            sys_classes_1 = self.classify(train_data,mode)
            t_acc, _ = evaluate(sys_classes_1, train_data)

            print("Train accuracy %.2f%%, Dev accuracy %.2f%%" % (t_acc, acc))

            # Let's not overfit too much :^)
            if t_acc >= 99.5:
                print('Training accuracy large enough, stopping training')
                break
            
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

    # output_dir = argv[2]
    output_files = {mode:open("%s/test.output.%s.txt" % ('results', mode),
                                encoding="utf-8",
                                mode="w")
                    for mode in MODES}

    print("Extracting features.")
    custom_extract_features(data)

    # Use the development set to tune the number of training epochs.
    epochs = {"basic":40, "averaged":20, "mira":40}
    # epochs = {"basic":0, "averaged":6, "mira":40}

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
