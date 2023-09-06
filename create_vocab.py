import json
import re
from collections import defaultdict
from tqdm import tqdm
import os


def get_vocab(data):
    """
    Given a list of strings, returns a dictionary of words mapping to their frequency 
    count in the data.
    """
    vocab = defaultdict(int)
    for line in data:
        for word in line:
            vocab[' '.join(list(word)) + ' </w>'] += 1
    return vocab
  
def get_stats(vocab):
    """ 
    Given a vocabulary (dictionary mapping words to frequency counts), returns a 
    dictionary of tuples representing the frequency count of pairs of characters 
    in the vocabulary.
    """
    pairs = defaultdict(int)
    for word, freq in vocab.items():
        symbols = word.split()
        for i in range(len(symbols)-1):
            pairs[symbols[i],symbols[i+1]] += freq
    return pairs
  
def merge_vocab(pair, v_in):
    """
    Given a pair of characters and a vocabulary, returns a new vocabulary with the 
    pair of characters merged together wherever they appear.
    """
    v_out = {}
    bigram = re.escape(' '.join(pair))
    p = re.compile(r'(?<!\S)' + bigram + r'(?!\S)')
    for word in v_in:
        w_out = p.sub(''.join(pair), word)
        v_out[w_out] = v_in[word]
    return v_out
  
def byte_pair_encoding(data, n):
    """
    Given a list of strings and an integer n, returns a list of n merged pairs
    of characters found in the vocabulary of the input data.
    """
    vocab = get_vocab(data)
    for i in tqdm(range(n)):
        pairs = get_stats(vocab)
        best = max(pairs, key=pairs.get)
        vocab = merge_vocab(best, vocab)
    return vocab




files = ["data/COCO_dataset/captions_train2017.json",
         "data/COCO_dataset/captions_val2017.json"]


all_tokens = set()
word_to_index = {}
index_to_word = {}



data = []

for file in files :

    with open(file, 'r') as f : 
        caption_dict = json.load(f)
        caption_dict = caption_dict["annotations"]

    for obj in caption_dict :
        caption = obj["caption"]
        #dont need spaces and special chars in BPE
        words = list(filter( lambda x : x not in [None, "", " "] , re.split("\s+|\W+", caption)))   
        #adding special chars to vocab      
        for x in list(filter(lambda x : x != " ", re.findall("\W{1}", caption))) :                  
            all_tokens.add(x)
        data.append(words)

bpe_pairs = sorted(byte_pair_encoding(data, 7500))

for word in bpe_pairs : 
    #word = word.replace('</w>', '')
    for token in word.split() :
        all_tokens.add(token)


word_to_index['<S>'] = 0                                      #sentence start token
word_to_index['</S>'] = 1                                     #sentence end token
word_to_index['<PAD>'] = 2                                    #Pad token

for i, word in enumerate((sorted(all_tokens))) :
    word_to_index[word] = i + 3

for i,j in word_to_index.items() :
    index_to_word[j] = i

save = {}

save["vocab_size"] = len(all_tokens) + 3
save["index_to_word"] = index_to_word
save["word_to_index"] = word_to_index



with open("vocab.json", "w") as f :
    json.dump(save, f, sort_keys= False, indent= 4)

print("Done !")