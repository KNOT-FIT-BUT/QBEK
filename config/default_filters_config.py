{  # All relative paths are related to this config file path.
    "result": "checkpoint_with_filters.ckpt",
    "language": "cze",
    # Is used for selecting lemmatizer for evaluation.
    # List of supported languages:
    #   cze for Czech
    #   eng for English
    #   ger for German
    #   fre for French
    # Path where new .ckpt with filters will be saved. (the training state will not be saved)
    "model": {
        "type": "independent",  # Variants: independent
        "checkpoint": "path_to_checkpoint.ckpt",  # path to trained model
    },
    "trials": 100,
    # number of triels when searching the best filters combination
    "validation": {  # dataset that will be used for evaluation
        "dataset": {
            "path": "../data/czech_library_small_single_sentence_val.txt",  # path to validation dataset
            # values > 0 activates multi process reading of dataset and the value determines number of subprocesses
            # that will be used for reading (the main process is not counted).
            # If == 0 than the single process processing is activated.
            "workers": 0,
            "add_title": True
        },
        "batch": 32  # batch size for validation
    },
    "transformers": {
        "tokenizer": "bert-base-multilingual-cased",
        # fast tokenizer that should be used (see https://huggingface.co/transformers/main_classes/tokenizer.html)

        # Cache where the transformers library will save the models.
        # Use when you want to specify specific path.
        "cache": None,

        "local_files_only": False   # If true only local files in cache will be used to load models and tokenizer
    },
    "filters": {
        # filters are not used during training process
        # Filters will be used during the test phase.
        # order of filters applications: n_best(score_threshold(max_span_size(results)))
        "n_best": "auto",  # Only N best (the ones with biggest score) spans are predicted.
        "max_span_size": "auto",  # No predicted span will have greater size (number of sub-word tokens) than this.
        "score_threshold": "auto",  # scores must be >= than this threshold
        #   You can use "auto" option to automatically find score_threshold or use manual value.
        #   For automatic filter finding a validation dataset is used on the very end of training process.
    },
    "gpu": {
        "allow": True,  # True activates computation on GPU (if GPU is available)
        "mixed_precision": False,  # true activates mixed precision training
    }
}
