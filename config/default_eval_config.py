{   # All relative paths are related to this config file path.
    "language": "cze",
    # Is used for selecting lemmatizer for evaluation.
    # List of supported languages:
    #   cze for Czech
    #   eng for English
    #   ger for German
    #   fre for French
    "metrics": ["multilabel_identification"],  # Variants: multilabel_identification
    "model": {
        "type": "independent",  # Variants: independent
        "checkpoint": "path_to_checkpoint.ckpt",  # path to trained model
    },
    "outputs": {
        "predictions": "predictions.tsv",  # file where the predictions should be saved
        "results": "evaluation.txt",  # file where evaluation should be saved
    },
    "validation": {
        "dataset": {
            "path": "../data/czech_library_small_single_sentence_val.txt",  # path to validation dataset
            # values > 0 activates multi process reading of dataset and the value determines number of subprocesses
            # that will be used for reading (the main process is not counted).
            # If == 0 than the single process processing is activated.
            "workers": 0,
            "add_title": True
        },

        "batch": 32    # batch size for testing
    },
    "transformers": {
        "tokenizer": "bert-base-multilingual-cased",  # fast tokenizer that should be used (see https://huggingface.co/transformers/main_classes/tokenizer.html)

        # Cache where the transformers library will save the models.
        # Use when you want to specify specific path.
        "cache": None,

        "local_files_only": False   # If true only local files in cache will be used to load models and tokenizer
    },
    "gpu": {
        "allow": True,  # True activates computation on GPU (if GPU is available)
        "mixed_precision": False,  # true activates mixed precision training
    }
}