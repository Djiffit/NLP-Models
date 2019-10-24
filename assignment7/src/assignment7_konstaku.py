"""
   Models and Algorithms in NLP Applications (LDA-H506)

   Starter code for Assignment 7: Lemmatization

   Miikka Silfverberg
"""

from sys import argv, stderr, stdout
from random import seed, shuffle
import os
import functools

import numpy as np
from scipy.special import logsumexp

seed(0)

from data import eval_lemmatizer, read_sigmorphon, write_sigmorphon, LANGUAGES
from paths import data_dir, results_dir

# Boundary tags at sequence boundaries.
INITIAL = "<INITIAL>"
FINAL = "<FINAL>"

# This toggles development vs final test mode. Set it to 0 when you
# want to use the entire training set and tag the test data. Note,
# that training on the entire training set may be slow.
DEVELOPMENT_MODE=1

# Control the number of training epochs for each language.
EPOCHS = {"finnish": 7, "german": 7, "spanish":11}

"""
    These are feature functions. Every feature function takes three
    arguments:

    * i, an index in [0,len(ex["TOKENS"] - 1)].
    * ex, an example.

    Add your own feature functions here.
"""

def bias(i,ex):
    return "BIAS"

def current_symbol(i,ex):
    return "WF[i]=%s" % ex["WF"][i]


def prev_symbol(i,ex):
    if i == 0:
        return "WF_prev[i]=%s" % '<START>'

    return "WF_prev[i]=%s" % ex["WF"][i - 1]

def next_symbol(i,ex):
    if i >= len(ex) - 1:
        return "WF_next[i]=%s" % '<END>'

    return "WF_next[i]=%s" % ex["WF"][i + 1]

def add_tags(i, ex):
    keys = set(ex['TAG']).difference(set(['pos']))
    return functools.reduce(lambda x, y: x + f'WF_{y}={ex["TAG"][y]} ', keys, ' ')

def seq_pos(i, ex):
    return f'WF_position[i]={i}'

def add_pos(i, ex):
    pos = ex['TAG']['pos']
    return f'WF_pos={pos}'

def get_id(f,i,ex,encoder):
    feature = f(i,ex)
    if not feature in encoder:
        encoder[feature] = len(encoder)
    return encoder[feature]

def add_per(i, ex):
    if not ex['TAG'].get('per', False):
        return f'WF_per=<UNK>'

    return f'WF_per={ex["TAG"]["per"]}'

def add_aspect(i, ex):
    if not ex['TAG'].get('aspect', False):
        return f'WF_aspect=<UNK>'

    return f'WF_aspect={ex["TAG"]["aspect"]}'

def add_tense(i, ex):
    if not ex['TAG'].get('tense', False):
        return f'WF_tense=<UNK>'

    return f'WF_tense={ex["TAG"]["tense"]}'

def add_finite(i, ex):
    if not ex['TAG'].get('finite', False):
        return f'WF_finite=<UNK>'

    return f'WF_finite={ex["TAG"]["finite"]}'

def add_mood(i, ex):
    if not ex['TAG'].get('mood', False):
        return f'WF_mood=<UNK>'

    return f'WF_mood={ex["TAG"]["mood"]}'


def add_num(i, ex):
    if not ex['TAG'].get('num', False):
        return f'WF_num=<UNK>'

    return f'WF_num={ex["TAG"]["num"]}'

def dist_to_end(i, ex):
    return f'WF_DIST_TO_END[i]={len(ex) - i - 1}'

def seq_len(i, ex):
    return f'WF_LEN={len(ex)}'

def suffix(i, ex):
    return f'WF_SUFFIX={ex["WF"][-3:]}'

    
def suffix_and_pos(i, ex):
    return f'WF_SUFFIX={ex["WF"][-3:]},POS={ex["TAG"]["pos"]}'

def last_letter(i, ex):
    return f'WF_LAST={ex["WF"][-1]}'

def next_three(i, ex):
    return f'WF_NEXT_THREE={ex["WF"][i:i+3]}'

def prev_three(i, ex):
    return f'WF_PREV_THREE={ex["WF"][i-3:i]}'

def curr_tri(i, ex):
    return f'WF_CURR_TRI={ex["WF"][i-1:i+1]}'

def extract_features(data):
    """
        This function extracts emission features. It uses the features
        functions listed in the feature_functions. 

        You should extend this list with your own feature functions.
    """
    feature_functions = [bias,
                        current_symbol, 
                        # prev_symbol, 
                        # next_symbol, 
                        add_tags, 
                        add_pos,
                        dist_to_end,
                        # suffix_and_pos,
                        suffix,
                        # add_num,
                        # add_mood,
                        # add_finite,
                        # add_aspect,
                        # add_tense,
                        prev_three,
                        next_three,
                        curr_tri,
                        last_letter,
                        seq_len, 
                        seq_pos
                        ]

    encoder = {}
    all_things = set([])
    for split in ["train","development","test"]:
        data_set = data[split]
        for ex in data_set:
            all_things = all_things.union(set(ex['TAG'].keys()))
            ex["FEATURES"] = [[get_id(f,i,ex,encoder) 
                               for f in feature_functions]
                              for i in range(len(ex["WF"]))]
    print(all_things)
    return encoder


def edit_distance(w1,w2):
    """
        Return the edit distance between the strings w1 and w2.

        You should provide a proper implementation for this function.
    """
    dp = np.array([[0] * (len(w2) + 1) for i in range(len(w1) + 1)])

    for y in range(len(w1) + 1):
        for x in range(len(w2) + 1):
            if y == 0:
                dp[0, x] = x
            elif x == 0:
                dp[y, 0] = y
            elif w1[y - 1] == w2[x - 1]:
                dp[y, x] = dp[y-1, x-1]
            else:
                dp[y, x] = min(dp[y-1, x-1], dp[y -1, x], dp[y, x- 1]) + 1
    return dp[-1, -1]

# Test edit distance
assert(edit_distance('pekka', 'pekka') == 0)
assert(edit_distance('peee', 'eeep') == 2)
assert(edit_distance('', '') == 0)
assert(edit_distance('a', 'a') == 0)
assert(edit_distance('weirdal', 'al') == 5)
assert(edit_distance('weirdal', '') == 7)
assert(edit_distance('', 'weirdal') == 7)
assert(edit_distance('abba', 'akka') == 2)
assert(edit_distance('entosyttooitt', 'endosytoosi') == 5)

def realign_training_data(data):
    """
        This function tries to improve the alignment between the word
        forms and lemmas in the training data.

        You should provide a proper implementation for this
        function. You only need to change ex["ALIGNED_LEMMA"].
    """
    output_characters = set()

    for ex in data["train"]:
        # Fill in your code for realigning word form and lemma here.

        if any([len(x) > 1 for x in ex['ALIGNED_LEMMA']]):
            word = []
            a = ex['ALIGNED_LEMMA']
            w = ex['WF']
            co = ''

            for ind, (align, wf) in enumerate(zip(a, w)):
                if len(align) > 1:
                    if ind < len(a) - 2 and a[ind + 1] == align[-1]: 
                        co = align[-1]
                        word += [a[ind][0]]
                    else:
                        word += [align]
                else:
                    if co:
                        word += [co + align]
                        co = ''
                    else:
                        word += [align]

            output_characters.update(word)
        output_characters.update(ex["ALIGNED_LEMMA"])

    data["output_characters"] = list(sorted(output_characters))

    return data

class StructuredPerceptron:
    """ 
        This is a simple structured classifier. It is initialized with
        a sets of output tags and a features.

        self.tags    - An encoder tag -> id number.
        self.id2tag  - A decoder id number -> tag.
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
            Recover the most probable character sequence from the trellis.

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
        features = [np.zeros(self.E['a'].shape[0]) for tok in ex["WF"]]
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

            No need to change this function.
        """
        # Get emission feature vectors for ex.        
        features = [np.zeros(self.E['a'].shape[0]) for tok in ex["WF"]]
        for i, ids in enumerate(ex["FEATURES"]):
            features[i][ids] = 1

        for i, i_features in enumerate(features):
            gold_tag = ex["ALIGNED_LEMMA"][i]
            sys_tag = sys_tags[i]

            self.E[gold_tag] += i_features
            self.E[sys_tag] -= i_features

        for tag1, tag2 in zip([INITIAL] + ex["ALIGNED_LEMMA"], \
                                  ex["ALIGNED_LEMMA"] + [FINAL]):
            self.T[self.tags[tag1],self.tags[tag2]] += 1

        for tag1, tag2 in zip([INITIAL] + sys_tags, sys_tags + [FINAL]):
            self.T[self.tags[tag1],self.tags[tag2]] -= 1

    def train(self,data,language):
        """
            This function trains the model. 

            No need to change this function.
        """
        for n in range(EPOCHS[language]):
            shuffle(data["train"])
            for i,ex in enumerate(data["train"]):
                sys_tags = self.classify_ex(ex)
                self.update_ex(sys_tags,ex)
                print("Example %u of %u" % (i+1,len(data["train"])),end="\r")
            print()
            print("Evaluating for epoch %u:" % (n+1),end=' ')
            acc, edit_dist = \
                eval_lemmatizer(self.classify(data["development"]),
                                data["development"],
                                edit_distance)
            print("Dev acc = %.2f, Dev avg edit dist = %.2f" % 
                  (100 * acc, edit_dist))

if __name__=="__main__":
    # Read training and test sets.
    print("Reading data (this may take a while).")
    data = read_sigmorphon(data_dir,use_mini_data=DEVELOPMENT_MODE)

    print("Building lemmatizers for: %s" % 
          ", ".join([l.title() for l in LANGUAGES]))
    print()

    for language in LANGUAGES:
        print("Building lemmatizer for %s." % language.title())
        realign_training_data(data[language])
        feature_encoder = extract_features(data[language])
        model = StructuredPerceptron(feature_encoder,
                                     data[language]["output_characters"])
        
        print("Training model.")
        model.train(data[language],language)

        print("Tagging, evaluating and storing development data.")
        dev_output = model.classify(data[language]["development"])
        write_sigmorphon(dev_output,data[language]["development"],
                         results_dir,language,"dev")
        acc, avg_edit_dist = eval_lemmatizer(dev_output, 
                                             data[language]["development"],
                                             edit_distance)
        print("dev acc = %.2f, dev avg edit dist: %.2f" % 
              (100 * acc, avg_edit_dist))

        if not DEVELOPMENT_MODE:
            print("Tagging, evaluating and storing test data.")
            test_output = model.classify(data[language]["test"])
            write_sigmorphon(test_output,data[language]["test"],
                             results_dir,language,"test")
            acc, avg_edit_dist = eval_lemmatizer(test_output, 
                                                 data[language]["test"],
                                                 edit_distance)
            print("test acc = %.2f, test avg edit dist: %.2f" % 
                  (100 * acc, avg_edit_dist))
        print()
        
