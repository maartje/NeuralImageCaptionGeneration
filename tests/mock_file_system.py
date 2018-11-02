mocked_file_storage = {
    'data/input/train.1.en' : [
        'Two dogs playing with a ball.',
        'A boy in black suit.',
        'Two men walking through the garden.',
        'Mountain covered with snow.',
        'A man reading the news paper.',
    ],
    'data/input/val.1.en' : [
        'A cat lying on the roof.',
        'A man waiting at the station.',
        'A woman in a red dress.'
    ],
    'data/input/test.1.en' : [
        'A mouse eating cheese.',
        'A green parc with a fauntain',
        'A cloudy sky.'
    ],
}

def mock_load(fpath):
    return mocked_file_storage[fpath]

def mock_save(data, fpath):
    mocked_file_storage[fpath] = data
    
def mock_glob(fpattern):
    return [fpattern]



