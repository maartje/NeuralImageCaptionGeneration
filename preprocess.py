from parse_config import get_configuration
    
def main():
    description = 'Generate vocabulary and indices vectors for captions'
    config, filepaths = get_configuration('preprocess', description = description)
    print (config)
    for fp in filepaths.items():
        print (fp)

    
if __name__ == "__main__":
    main()
