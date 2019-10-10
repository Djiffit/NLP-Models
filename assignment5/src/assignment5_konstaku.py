"""
   Models and Algorithms in NLP Applications (LDA-H506)

   Starter code for Assignment 5: HMM

   Miikka Silfverberg
"""

from sys import argv, stderr, stdout
from random import seed, shuffle
import os

import numpy as np

seed(0)
np.random.seed(0)

from data import eval_ner, read_conll_ner
from paths import data_dir, results_dir

# Boundary tags at sentence boundaries.
INITIAL = "<INITIAL>"
FINAL = "<FINAL>"

# Unknown token.
UNK = "<UNK>"

class HMM:
    """ 
        This is a simple HMM model. It is initialized with the sets of
        word forms, NER tags, and POS tags in the training data.

        alpha              - Smoothing term.
        vocab              - Vocabulary translates word forms to index numbers.
        tags               - Translates NER tags to index numbers.
        id2tags            - Translates index numbers to NER tags.
        pos_tags           - Translates POS tags to index numbers.
        emission_counts    - Emission counts for NER tags and word forms.
        transition_counts  - Transition counts for NER tag pairs.
        pos_emission_counts- Emission counts for NER tags and POS tags.
    """
    def __init__(self, vocab, tags, pos_tags):
        # Alpha for Laplace smoothing.
        self.alpha = .01

        self.vocab = {wf:i for i, wf in enumerate(vocab + [UNK])}
        self.tags = {tag:i for i, tag in enumerate([INITIAL] + tags + [FINAL])}
        self.id2tag = {i:tag for tag, i in self.tags.items()}
        self.pos_tags = {pos:i for i,pos in enumerate(pos_tags + [UNK])}

        self.emission_counts = np.zeros((len(self.tags), len(self.vocab)))
        self.transition_counts = np.zeros((len(self.tags), len(self.tags)))
        self.pos_emission_counts =np.zeros((len(self.tags),len(self.pos_tags)))

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

            It is your job to implement this function properly.
        """

        # We first reserve a trellis for probabilities and an index trellis.
        trellis = np.full((len(ex["TOKENS"]) + 2,len(self.tags)),-float('inf'))
        indices = -np.ones((len(ex["TOKENS"]) + 2,len(self.tags)))
        
        trellis[0,self.tags[INITIAL]] = 0

        # Iteratively compute trellis probabilities pi(i,y) and update
        # the index trellis.
        for i, token in enumerate(ex["TOKENS"]):
            # Replace unknown tokens in test time with UNK.
            if not token in self.vocab:
                token = UNK

            pos = ex["POS TAGS"][i]
            # Replace unknown POS tags in test time with UNK.
            if not pos in self.pos_tags:
                pos = UNK

            # Compute pi(i,y) for each possible tag y. You will need
            # to use self.E and self.T. As a starred exercise, you can
            # add probabilitites from self.E_pos.s
            for tag in self.tags:
                # Words can't receive the inital or final state tag.
                if tag in [INITIAL,FINAL]:
                    continue

                tag_index = self.tags[tag]
                token_index = self.vocab[token]

                # Correctly compute all entries in pi_candidates.
                # Replace the lines below by your own code.

                candidates = [prob + self.E[tag_index, token_index] + self.T[ind, tag_index] + self.E_pos[tag_index, self.pos_tags[pos]] for ind, prob in enumerate(trellis[i])]
                trellis[i+1][self.tags[tag]] = np.max(candidates)
                indices[i+1][self.tags[tag]] = np.argmax(candidates)

        # Correctly compute all entries in pi_final_candidates.
        # Replace the lines below by your own code.

        pi_final_candidates = [prob + self.T[ind, self.tags[FINAL]] for ind, prob in enumerate(trellis[i + 1])]
        trellis[len(ex["TOKENS"]) + 1,self.tags[FINAL]] = np.max(pi_final_candidates)
        indices[len(ex["TOKENS"]) + 1,self.tags[FINAL]] = np.argmax(pi_final_candidates)

        return self.recover_tags(indices,indices.shape[0] - 2,
                                 int(indices[-1,self.tags[FINAL]]))

    def classify(self,data):
        """f
            This function classifies a data set. 

            No need to change this function.
        """
        return [self.classify_ex(ex) for ex in data]

    def update_ex(self,ex):
        """
            This function performs HMM updates for one labeled
            example.

            It is your job to update emission and transition counts properly.
            
            As a starred exercise, you can also update POS emission counts.
        """
        for i, (token, tag) in enumerate(zip(ex["TOKENS"],ex["TAGS"])):
            self.emission_counts[self.tags[tag], self.vocab[token]] += 1

        for tag1, tag2 in zip([INITIAL] + ex["TAGS"], ex["TAGS"] + [FINAL]):
            self.transition_counts[self.tags[tag1], self.tags[tag2]] += 1

        for pos, tag in zip(ex["POS TAGS"],ex["TAGS"]):
            self.pos_emission_counts[self.tags[tag], self.pos_tags[pos]] += 1

    def train(self,train_data):
        """
            This function trains the model. 

            It is your job to compute the emission and transition
            probability matrices E and T.
            
            As a starred assignment, you can compute E_pos which
            represents POS emissions.
        """
        for ex in train_data:
            self.update_ex(ex)

        # Smooth emission counts, normalize and go over to log space.


        self.E = self.emission_counts + self.alpha
        self.E = np.log((self.E.T / np.sum(self.E, axis=1)).T)

        # self.E = np.zeros(self.emission_counts.shape)
        # Smooth transition counts, normalize and go over to log space.
        self.T = self.transition_counts + self.alpha
        self.T = np.log((self.T.T / np.sum(self.T, axis=1)).T)

        # Smooth POS counts, normalize and go over to log space.
        self.E_pos = self.pos_emission_counts + self.alpha
        self.E_pos = np.log((self.E_pos.T / np.sum(self.E_pos, axis=1)).T)

if __name__=="__main__":
    # Read training and test sets.
    print("Reading data (this may take a while).")
    data, vocab, tags, pos_tags = read_conll_ner(data_dir)

    for alpha in [ 0.0001, 0.01, .1, 1 ]:
        model = HMM(vocab,tags, pos_tags)
        model.alpha = alpha
        print(f'--------------- Training model with alpha {alpha} -----------------------')
        model.train(data["train"])

        print("Tagging and evaluating development data.")
        recall, precision, fscore = eval_ner(model.classify(data["development"]),
                                            data["development"])
        print("Recall: %.2f" % (100 * recall))
        print("Precision: %.2f" % (100 * precision))
        print("F1-Score: %.2f" % (100 * fscore))
