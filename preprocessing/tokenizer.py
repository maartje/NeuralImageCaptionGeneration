from mosestokenizer import MosesTokenizer, MosesDetokenizer

def preprocess(sentence): 
    return sentence.lower().strip()

def sentence2tokens(sentence):
    sentence_lc = preprocess(sentence)
    with MosesTokenizer('en') as tokenize:
        tokens = tokenize(sentence_lc)
    return tokens

def tokens2sentence(tokens):
    with MosesDetokenizer('en') as detokenize:
        sentence = detokenize(tokens) 
    # We do not restore capitalization, instead we lowercase reference sentences
    return sentence


