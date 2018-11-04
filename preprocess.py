from parse_config import get_configuration
from preprocessing.textmapper import TextMapper
from debug_helpers import format_duration

import torch
from datetime import datetime

def preprocess(filepaths, config):    
    start_time = datetime.now()
    mapper = build_and_save_vocabulary(filepaths, config, start_time)
    build_and_save_caption_vectors(filepaths, mapper, start_time)

def build_and_save_vocabulary(filepaths, config, start_time):
    print(f'\n({format_duration(start_time, datetime.now())}) Start building vocabulary ...')
    sentences_train = read_lines_multiple_files(filepaths['captions_train'])         
    mapper = TextMapper()
    mapper.build(sentences_train, config['min_occurences'])
    torch.save(mapper, filepaths['vocab'])
    duration_str = format_duration(start_time, datetime.now())
    print(f"({duration_str})    Saved vocabulary file at {filepaths['vocab']}")
    print(f'({format_duration(start_time, datetime.now())}) Finished building vocabulary ...')
    return mapper

def build_and_save_caption_vectors(filepaths, mapper, start_time):
    duration_str = format_duration(start_time, datetime.now())
    print(f'\n({duration_str}) Start building index vectors from descriptions ...')
    fpaths = filepaths['captions_train'] + filepaths['captions_val'] + filepaths['captions_test']
    fpaths_out = filepaths['caption_vectors_train'] + filepaths['caption_vectors_val'] + filepaths['caption_vectors_test']
    for (fpath, fpath_out) in zip(fpaths, fpaths_out):
        sentence_vectors = [mapper.sentence2indices(sentence) for sentence in read_lines(fpath)]
        torch.save(sentence_vectors, fpath_out)
        duration_str = format_duration(start_time, datetime.now())
        print(f'({duration_str})    Saved indices file at {fpath_out}')
    duration_str = format_duration(start_time, datetime.now())
    print(f'({duration_str}) Finished building index vectors from descriptions ...')

def read_lines_multiple_files(fpaths):
    for fpath in fpaths:
        for sentence in read_lines(fpath):
            yield sentence
            
def read_lines(fpath):
    with open(fpath, 'r') as lines:
        for line in lines:
            yield line.strip()

def main():
    description = 'Generate vocabulary and indices vectors for captions'
    filepaths, config = get_configuration('preprocess', description = description)
    preprocess(filepaths, config)

    
if __name__ == "__main__":
    main()
