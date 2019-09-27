import csv
import sys
import os

import numpy as np

UDHEADER=["ID","FORM","LEMMA","UPOS","XPOS","FEATS","HEAD","DEPREL","DEPS",
          "MISC"]
SEMEVALHEADER = ['ID','SENTIMENT','BODY']

def read_conllu(filename):
    """ 
    Read a CoNLL-U file from @filename into a list of sentences. Each
    sentence is a list of tokens. Each token is a dictionary with
    keys:

      ID     - ID number of the token in the sentence.
      FORM   - Orthographic representation of the token, e.g. "cats".
      LEMMA  - The lemma, e.g. "cat".
      UPOS   - The Universal Dependencies 2.0 POS tag.
      XPOS   - Language specific POS tag.
      FEATS  - Morphological features.
      HEAD   - The id of the head word of this token (0 if this is 
               the head of the sentence).
      DEPREL - The dependency label between this word and its head 
               ('root' if this is the head of the sentence).
      DEPS   - Enhanced dependencies.
      MISC   - Other annotations.

      For example,

      {'ID': '1', 'FORM': 'Al', 'LEMMA': 'Al', 'UPOS': 'PROPN', 'XPOS': 'NNP', 
       'FEATS': 'Number=Sing', 'HEAD': '0', 'DEPREL': 'root', 'DEPS': '_', 
       'MISC': 'SpaceAfter=No'}
    """
    data = [[]]

    with open(filename,encoding="utf-8") as udfile:
        csvreader = csv.reader(filter(lambda x:x == '' or x[0]!='#', udfile),
                               delimiter='\t',
                               quoting=csv.QUOTE_NONE)
        for i, fields in enumerate(csvreader):
            if fields and len(fields) != len(UDHEADER):
                raise SyntaxError('Incorrect field count', 
                                  (filename, i, None, None)) 
            if fields == []:
                data.append([])
            else:
                data[-1].append(dict(zip(UDHEADER,fields)))

    return data
                    
def read_semeval(filename):
    """
    Read a list of tweets with sentiment labels from @sefilename. Each
    tweet is a dictionary with keys:
 
    ID        - ID number of tweet.
    SENTIMENT - Sentiment label for this tweet.
    BODY      - List of tokens of this tweet.

    """
    data = []

    with open(filename,encoding="utf-8") as sefile:
        csvreader = csv.reader(sefile, delimiter='\t', quoting=csv.QUOTE_NONE)
        for i, fields in enumerate(csvreader):
            if fields and len(fields) != len(SEMEVALHEADER):
                raise SyntaxError('Incorrect field count', 
                                  (filename, i, None, None)) 
            tweet = dict(zip(SEMEVALHEADER,fields))
            tweet['ORIG_BODY'] = tweet['BODY']
            data.append(tweet)
    return data

def read_semeval_datasets(data_dir):
    data = {}
    for data_set in ["training","development.input","development.gold",
                     "test.input","test.gold"]:
        data[data_set] = read_semeval(os.path.join(data_dir,"%s.txt" % data_set)) 
    return data


def evaluate(sys_classes, gold_data):
    classes = set(sys_classes + [ex["SENTIMENT"] for ex in gold_data])
    cgncy_table = {(c1,c2):0 for c1 in classes for c2 in classes}
    gold_table = {c:0 for c in classes}
    sys_table = {c:0 for c in classes}

    for sys_c, gold_ex in zip(sys_classes,gold_data):
        gold_c = gold_ex["SENTIMENT"]
        cgncy_table[(sys_c,gold_c)] += 1
        gold_table[gold_c] += 1
        sys_table[sys_c] += 1
        
    fscores = {}

    for c in classes:        
        c_recall = (cgncy_table[(c,c)] * 1.0 / gold_table[c] 
                    if gold_table[c] > 0
                    else 0)
        c_precision = (cgncy_table[(c,c)] * 1.0 / sys_table[c]
                       if sys_table[c] > 0
                       else 0)
        fscores[c] = (200 * (c_recall * c_precision) / (c_recall + c_precision)
                      if c_recall + c_precision > 0
                      else 0)
                      
    accuracy = (sum([cgncy_table[(c,c)] for c in classes]) * 100.0 / 
                len(sys_classes))

    return accuracy, fscores

def write_semeval(data,output,output_file):
    for ex, klass in zip(data,output):
        print("%s\t%s\t%s" % (ex["ID"], klass, ex["ORIG_BODY"]),
              file=output_file)

def get_class(distribution):
    return list(distribution.keys())[np.argmax(list(distribution.values()))]

if __name__=="__main__":
    # Check that we don't crash on reading.
    read_semeval('%s/semeval/training.txt' % sys.argv[1])
    read_semeval('%s/semeval/development.input.txt' % sys.argv[1])
    read_semeval('%s/semeval/development.gold.txt' % sys.argv[1])
    read_semeval('%s/semeval/test.input.txt' % sys.argv[1])
    read_semeval('%s/semeval/test.gold.txt' % sys.argv[1])

    read_conllu('%s/ud/en-ud-train.txt' % sys.argv[1])
    read_conllu('%s/ud/en-ud-dev.input.txt' % sys.argv[1])
    read_conllu('%s/ud/en-ud-dev.gold.txt' % sys.argv[1])
    read_conllu('%s/ud/en-ud-test.input.txt' % sys.argv[1])
    read_conllu('%s/ud/en-ud-test.gold.txt' % sys.argv[1])

