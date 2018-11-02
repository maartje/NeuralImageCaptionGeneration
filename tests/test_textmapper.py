"""
Tests for the text mapper that maps words to indices and indices to words
"""

import unittest
from preprocessing.textmapper import TextMapper
import warnings

class TestTextMapper(unittest.TestCase):

    def setUp(self):
        warnings.simplefilter("ignore")
        sentences = ['Hello world!', 'Hello Foo.']
        self.mapper = TextMapper()
        self.mapper.build(sentences)
        
    def test_sentence2indices_inserts_special_tokens(self):
        sentence = 'Hello foo bar'
        indices = self.mapper.sentence2indices(sentence)
        SOS_index = self.mapper.vocab.word2index[self.mapper.SOS]
        EOS_index = self.mapper.EOS_index()
        UNKNOWN_index = self.mapper.vocab.word2index[self.mapper.UNKNOWN]
        
        self.assertEqual(SOS_index, indices[0])
        self.assertEqual(UNKNOWN_index, indices[-2])
        self.assertEqual(EOS_index, indices[-1])

    def test_indices2sentence_inverts_sentence2indices(self):
        sentence = 'Hello foo bar!'
        indices = self.mapper.sentence2indices(sentence)
        sentence_out = self.mapper.indices2sentence(indices)
        
        self.assertEqual('hello foo!', sentence_out)

if __name__ == '__main__':
    unittest.main()


