"""
Tests for the vocabulary containing the index and inverted index
"""

import unittest
from preprocessing.vocabulary import Vocabulary

class TestVocabulary(unittest.TestCase):

    def test_build(self):
        words = ['hello', 'good', 'world', ',', 'hello', 'hello', 'bad', 'world']
        predefined = ['SOS', 'EOS', 'UNKNOWN']
        min_occurrence = 2
        
        vocab = Vocabulary()
        vocab.build(words, predefined, min_occurrence)

        # one-to-one mapping between word tokens and indexes         
        self.assertEqual('hello', vocab.index2word[vocab.word2index['hello']])

        # predefined words in index head
        self.assertEqual(0, vocab.word2index['SOS'])
        self.assertEqual('EOS', vocab.index2word[1])

        # words with low frequency not in idexes
        self.assertFalse('good' in vocab.word2index)

if __name__ == '__main__':
    unittest.main()


