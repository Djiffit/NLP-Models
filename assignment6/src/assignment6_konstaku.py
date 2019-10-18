"""
   Models and Algorithms in NLP Applications (LDA-H506)

   Starter code for Assignment 6: Structured Perceptron

   Miikka Silfverberg
"""

from sys import argv, stderr, stdout
from random import seed, shuffle
import os

import nltk
import numpy as np
try:
    from scipy.special import logsumexp
except (NameError, ImportError):
    from scipy.misc import logsumexp

seed(0)

from data import eval_ner, read_conll_ner, read_gazetteers, write_conll_ner
from paths import data_dir, results_dir

# Boundary tags at sentence boundaries.
INITIAL = "<INITIAL>"
FINAL = "<FINAL>"

# This toggles development vs final test mode. Set it to 0 when you
# want to use the entire training set and tag the test data. Note,
# that training on the entire training set may be (really) slow.
DEVELOPMENT_MODE=1

EPOCHS = 10

"""
    These are feature functions. Every feature function takes three
    arguments:

    * i, an index in [0,len(ex["TOKENS"] - 1)].
    * ex, an example.
    * gazetteers, a collection of gazetteers.

    Add your own feature functions here.
"""

def bias(i,ex,gazetteers):
    return "BIAS"

def word(i,ex,gazetteers):
    return "WORD=%s" % ex["TOKENS"][i]

def first_upper(i, ex, gazetteers):
    tok = ex["TOKENS"][i]
    return f'UPPER={tok[0].isupper()}'

def all_upper(i, ex, gazetteers):
    tok = ex["TOKENS"][i]
    return f'ALL_UPPER={all([c.isupper() for c in tok])}'

def pos_tag(i, ex, gazetteers):
    pos = ex['POS TAGS'][i]
    return f'POS={pos}'

def chunk(i, ex, gazetteers):
    chunk = ex['CHUNKS'][i]
    return f'POS={chunk}'

def prev_word(i, ex, gazetteers):
    if i == 0:
        return f'PREV=<START>'
    tok = ex["TOKENS"][i - 1]
    
    return f'PREV={tok}'


def next_word(i, ex, gazetteers):
    if i == len(ex['TOKENS']) - 1:
        return f'NEXT=<END>'
    tok = ex["TOKENS"][i + 1]
    
    return f'NEXT={tok}'


# Shouldn't put multiple things into one now that I look how the features are built but oh well ...
def suffixes(i, ex, gazetteers):
    tok = ex["TOKENS"][i]
    if len(tok) < 3:
        return f'SUFF_1={tok[-1]} SUFF_2=<SHORT> SUFF_3=<SHORT>'

    return f'SUFF_1={tok[-1]} SUFF_2={tok[-2:]} SUFF_3={tok[-3:]}'

def gazetteer(i, ex, gazetteers):
    tok = ex["TOKENS"][i]
    is_geo, is_org, is_person = [tok in col for key, col in gazetteers.items()]

    return f'GEO_NAME={is_geo} ORG_NAME={is_org} PERSON_NAME={is_person} '

def prev_entity(i, ex, gazetteers):
    if i == 0:
        return f'PREV_GEO={False} PREV_ORG={False} PREV_PERSON={False} ' 
    tok = ex["TOKENS"][i - 1]
    is_geo, is_org, is_person = [tok in col for key, col in gazetteers.items()]

    return f'PREV_GEO={is_geo} PREV_ORG={is_org} PREV_PERSON={is_person}' 

def next_entity(i, ex, gazetteers):
    if i == len(ex['TOKENS']) - 1:
        return f'NEXT_GEO={False} NEXT_ORG={False} NEXT_PERSON={False} ' 
    tok = ex["TOKENS"][i + 1]
    is_geo, is_org, is_person = [tok in col for key, col in gazetteers.items()]

    return f'NEXT_GEO={is_geo} NEXT_ORG={is_org} NEXT_PERSON={is_person}' 

def prefixes(i, ex, gazetteers):
    tok = ex["TOKENS"][i]
    if len(tok) < 3:
        return f'PREF_1={tok[0]} PREF_2=<SHORT> PREF_3=<SHORT>'

    return f'PREF_1={tok[0]} PREF_2={tok[:2]} PREF_3={tok[:3]}'


# Extract feature and transform it into an ID number.
def get_id(f,i,ex,gazetteers,encoder):
    feature = f(i,ex,gazetteers)
    if not feature in encoder:
        encoder[feature] = len(encoder)
    return encoder[feature]

def extract_features(data,gazetteers):
    """
        This function extracts emission features. It uses the features
        functions listed in the feature_functions. You should extend
        this list with your own feature functions.
    """
    feature_functions = [bias,
                        first_upper,
                        all_upper,
                        next_word,
                        pos_tag,
                        chunk,
                        prev_word,
                        prev_entity,
                        next_entity,
                        suffixes,
                        prefixes,
                        gazetteer,
                        word]

    encoder = {}
    for data_set in data.values():
        for ex in data_set:
            ex["FEATURES"] = [[get_id(f,i,ex,gazetteers,encoder) 
                               for f in feature_functions]
                              for i in range(len(ex["TOKENS"]))]
    return encoder

class StructuredPerceptron:
    """ 
        This is a simple HMM model. It is initialized with the sets of
        word forms, NER tags, and POS tags in the training data.

        self.tags    - An encoder tag -> id number.
        self.td2tag  - A decoder id numbef -> tag.
        self.E       - Emission feature weights.
        self.T       - Transition feature weights.
    """
    def __init__(self, feature_encoder, tags):
        self.tags = {tag:i for i, tag in enumerate([INITIAL] + tags + [FINAL])}
        self.id2tag = {i:tag for tag,i in self.tags.items()}

        self.E = {tag:np.zeros(len(feature_encoder))
                  for tag in self.tags}        

        self.T = np.zeros((len(self.tags),len(self.tags)))

    def recover_tags(self,idxs,i,y):
        """
            Recover the most probable NER tag sequence from the trellis.

            No need to change this function.
        """
        return ([] if i == 0 else 
                self.recover_tags(idxs,i-1, int(idxs[i,y])) + [self.id2tag[y]])

    def classify_ex(self,ex):
        """
            Classify one example using the model.

            No need to change this function.
        """
        # Get emission feature vectors for ex.

        features = [np.zeros(self.E['O'].shape[0]) for tok in ex["TOKENS"]]
        for i, ids in enumerate(ex["FEATURES"]):
            features[i][ids] = 1

        trellis = np.full((len(features) + 2,len(self.tags)),
                          -float('inf'))
        indices = -np.ones((len(features) + 2,len(self.tags)))
        trellis[0,self.tags[INITIAL]] = 0

        for i, i_features in enumerate(features):
            for tag in self.tags:
                if tag in [INITIAL,FINAL]:
                    continue
                scores_tag = (trellis[i] + self.T[:,self.tags[tag]] + 
                              self.E[tag] @ i_features)
                trellis[i+1][self.tags[tag]] = np.max(scores_tag)
                indices[i+1][self.tags[tag]] = np.argmax(scores_tag)

        scores_final = trellis[len(features)] + self.T[:,self.tags[FINAL]]
        trellis[len(features) + 1,self.tags[FINAL]] = np.max(scores_final)
        indices[len(features) + 1,self.tags[FINAL]] = np.argmax(scores_final)

        return self.recover_tags(indices,indices.shape[0] - 2,
                                 int(indices[-1,self.tags[FINAL]]))

    def classify(self,data):
        """
            This function classifies a data set. 

            No need to change this function.
        """
        return [self.classify_ex(ex) for ex in data]

    def update_ex(self,sys_tags,ex):
        """
            This function performs structured perceptron updates for
            one labeled example.

            It is your job to update emission and transition feature
            weights correctly.
        """
        # Get emission feature vectors for ex.

        features = [np.zeros(self.E['O'].shape[0]) for tok in ex["TOKENS"]]
        for i, ids in enumerate(ex["FEATURES"]):
            features[i][ids] = 1

        for i, i_features in enumerate(features):
            gold_tag = ex["TAGS"][i]
            sys_tag = sys_tags[i]

            self.E[gold_tag] += i_features
            self.E[sys_tag] -= i_features
            

        for tag1, tag2 in zip([INITIAL] + ex["TAGS"], ex["TAGS"] + [FINAL]):
            t1 = self.tags[tag1]
            t2 = self.tags[tag2]
            self.T[t1, t2] += 1


        for tag1, tag2 in zip([INITIAL] + sys_tags, sys_tags + [FINAL]):
            t1 = self.tags[tag1]
            t2 = self.tags[tag2]
            self.T[t1, t2] -= 1

    def train(self,train_data):
        """
            This function trains the model. 

            No need to change this function.
        """

        for n in range(EPOCHS):
            shuffle(data["train"])
            for i,ex in enumerate(train_data):
                sys_tags = self.classify_ex(ex)
                self.update_ex(sys_tags,ex)
            _, _, fscore = eval_ner(self.classify(data["development"]),
                                    data["development"])
            print("Epoch %u : Dev f-score: %.2f" % (n+1,100 * fscore))

if __name__=="__main__":
    # Read training and test sets.
    print("Reading data (this may take a while).")
    data, vocab, tags, _ = read_conll_ner(data_dir)
    print("Reading gazetteers.")
    gazetteers = read_gazetteers(data_dir)
    if DEVELOPMENT_MODE:
        data['train'] = data['train'][:1000]
    feature_encoder = extract_features(data,gazetteers)

    model = StructuredPerceptron(feature_encoder,tags)

    print("Training model.")
    model.train(data["train"])

    print("Tagging, evaluating and storing test data.")
    test_sys = model.classify(data["test"])
    write_conll_ner(test_sys,data["test"],results_dir)
    recall, precision, fscore = eval_ner(test_sys, data["test"])
    print("Recall: %.2f" % (100 * recall))
    print("Precision: %.2f" % (100 * precision))
    print("F1-Score: %.2f" % (100 * fscore))

    corr_word = 'Espen'
    e_class = 'B-PER'
    wrong_word = 'Bali'
    w_exp = 'B-LOC'
    w_got = 'O'

    ex = {'TOKENS': ['7.', corr_word, 'Bredesen'], 'POS TAGS': ['NNP', 'NNP', 'NNP'], 'CHUNKS': ['I-NP', 'I-NP', 'I-NP']}

    feature_functions = [bias,
                    first_upper,
                    all_upper,
                    next_word,
                    pos_tag,
                    chunk,
                    prev_word,
                    prev_entity,
                    next_entity,
                    suffixes,
                    prefixes,
                    gazetteer,
                    word]

    print('TRANSITION FIRST', model.T[model.tags['O'],model.tags['B-PER']])
    for f in feature_functions:
        try:
            print(f.__name__, model.E['B-PER'][feature_encoder[f(1, ex, gazetteers)]])
        except:
            pass

    ex = {'TOKENS': [',', wrong_word, '1996-12-07'], 'POS TAGS': ['NNP', 'NNP', 'NNP'], 'CHUNKS': ['B-NP', 'B-NP', 'B-NP']}

    print('TRANSITION SECOND, correct', model.T[model.tags['O'],model.tags['B-LOC']])
    for f in feature_functions:
        try:
            feats = feature_encoder[f(1, ex, gazetteers)]
            print('EXPECTED CLASS', f.__name__, model.E[w_exp][feats])
        except:
            pass

    print('TRANSITION THIRD, wrong', model.T[model.tags['O'],model.tags['O']])
    for f in feature_functions:
        try:
            feats = feature_encoder[f(1, ex, gazetteers)]
            print('WRONG CLASS', f.__name__, model.E[w_got][feats])
        except:
            pass
        



