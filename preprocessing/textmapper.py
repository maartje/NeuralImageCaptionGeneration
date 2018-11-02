from preprocessing.tokenizer import sentence2tokens, tokens2sentence
from preprocessing.vocabulary import Vocabulary
from itertools import chain

class TextMapper:

    def __init__(self):
        self.vocab = Vocabulary()
        self.EOS = "EOS"
        self.SOS = "SOS"
        self.UNKNOWN = "UNKNOWN"
        self.PAD = "PAD"
    
    def build(self, sentences, min_occurence = 1):
        sentences_split = (sentence2tokens(sentence) for sentence in sentences)
        words = chain.from_iterable(sentences_split)
        self.vocab.build(words, [self.SOS, self.EOS, self.PAD, self.UNKNOWN], min_occurence)

    def PAD_index(self):
        return self.vocab.word2index[self.PAD]

    def EOS_index(self):
        return self.vocab.word2index[self.EOS]

    def SOS_index(self):
        return self.vocab.word2index[self.SOS]
    
    def sentence2indices(self, sentence):
        return self.tokens2indices(sentence2tokens(sentence))

    def indices2sentence(self, indices):
        return tokens2sentence(self.indices2tokens(indices, True))

    def tokens2indices(self, tokens):
        indices = [self.token2index(t) for t in tokens]
        return [self.token2index(self.SOS)] + indices + [self.token2index(self.EOS)] # add SOS, EOS

    def token2index(self, t):
        return self.vocab.word2index.get(t, self.vocab.word2index[self.UNKNOWN])

    def remove_predefined_indices(self, indices):
        if self.EOS_index() in indices: # take prefix up to EOS
            EOS_sentence_index = indices.index(self.EOS_index())
            indices_cut = indices[:EOS_sentence_index + 1] 
        else:
            indices_cut = indices
        predefined = [
           self.token2index(self.EOS), 
           self.token2index(self.SOS), 
           self.token2index(self.UNKNOWN),
           self.token2index(self.PAD),
        ]
        return [
            i for i in indices_cut if not i in predefined # remove SOS, EOS, UNKNOWN, PAD
        ] 
    
    def indices2tokens(self, indices, remove_predefined_tokens):
        if remove_predefined_tokens:
            indices = self.remove_predefined_indices(indices)
        return [self.index2token(i) for i in indices] 

    def index2token(self, i):
        return self.vocab.index2word[i]
        


