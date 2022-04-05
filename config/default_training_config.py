{   # All relative paths are related to this config file path.
    "name": "Example experiment",   # name of experiment (could be used for file naming)
    "fixed_seed": None,
    "model": {
        "type": "independent",  # Variants: independent
        "transformer": "bert-base-multilingual-cased",  # transformer model that will be used (see https://huggingface.co/models)
        "init_weights": True,  # Uses pretrained weights for transformer part. If False uses random initialization.
        "freeze_transformer": False  # freezes the transformer part of the model except the input embeddings
    },
    "transformers": {
        "tokenizer": "bert-base-multilingual-cased",  # fast tokenizer that should be used (see https://huggingface.co/transformers/main_classes/tokenizer.html)

        # Cache where the transformers library will save the models.
        # Use when you want to specify specific path.
        "cache": None,

        "local_files_only": False   # If true only local files in cache will be used to load models and tokenizer
    },
    "outputs": {
        "checkpoints": {
            "dir": "checkpoints",  # directory where the checkpoints will be saved (one checkpoint for each validation)
            "save_top_k": 1
            # We check whether to save a checkpoint after each validation.
            # Description from pytorch lightning:
            #   if save_top_k == k, the best k models according to the quantity monitored will be saved.
            #   if save_top_k == 0, no models are saved.
            #   if save_top_k == -1, all models are saved.
            #   Please note that the monitors are checked every period epochs.
            #   if save_top_k >= 2 and the callback is called multiple times inside an epoch,
            #   the name of the saved file will be appended with a version count starting with v1.
        },
        "results": "results",  # folder where results on test set will be saved (None means no results saving)
        "logs": "logs",  # folder where log files will be saved (each log file is saved in folder with experiment's name)
    },
    "training": {
        "dataset": {
            "path": "../data/czech_library_small_single_sentence_val.txt",  # path to training dataset
            # values > 0 activates multi process reading of dataset and the value determines number of subprocesses
            # that will be used for reading (the main process is not counted).
            # If == 0 than the single process processing is activated.
            "workers": 0,
            "prep_workers": 0,
            "add_title": True
        },

        "batch": 8,    # batch size for training
        "max_grad_norm": 5.0,
        # gradient normalization (see https://pytorch.org/docs/stable/generated/torch.nn.utils.clip_grad_norm_.html)
        # None and zero deactivates gradient normalization

        # Number of batches till the optimization step is done.
        # Values greater than one activates gradient accumulation.
        "accu_grad": 1,

        # maximum number of training epochs
        "max_epochs": 200,

        # maximum number of training steps
        # on resuming the resumed update steps are counted too
        # this value can be utilized by scheduler
        "max_steps": 500,

        # The training process will be stopped after X subsequent validations that shows no improvement.
        # None deactivates early stopping.
        "early_stopping": 4,

        "optimizer": {  # Adam with weight decay is used.
            "learning_rate": 1e-05,
            "weight_decay": 1e-2,   # weight decay for optimizer (see https://pytorch.org/docs/stable/optim.html?highlight=adamw#torch.optim.AdamW)
        },
        "scheduler": {
            "type": "linear",  # Options: None, linear, cosine, constant (see https://huggingface.co/transformers/main_classes/optimizer_schedules.html#schedules)
            "scheduler_warmup_proportion": 0.25,  # scheduler_warmup_proportion * max_steps is number of steps for warmup
        },
        "online_weighting": True    # loss will use online calculated class weights
    },
    "validation": {
        "dataset": {
            "path": "../data/czech_library_small_single_sentence_val.txt",  # path to validation dataset
            # values > 0 activates multi process reading of dataset and the value determines number of subprocesses
            # that will be used for reading (the main process is not counted).
            # If == 0 than the single process processing is activated.
            "workers": 0,
            "prep_workers": 0,
            "add_title": True
        },

        "steps": 50,  # After each X steps [batches] the validation will be performed.
        "batch": 8    # batch size for validation
    },
    "resume": {
        # resumes training from a given checkpoint
        # If resuming from a mid-epoch checkpoint, training will start from the beginning of the next epoch.

        "activate": False,  # True resumes training
        "checkpoint": None,  # path to checkpoint

        # False means that you want to resume whole training process
        # (scheduler, optimizer, the walk trough dataset ...)  from checkpoint.
        # True resumes just the trained model and the training will start again.
        "just_model": False,
    },
    "gpu": {
        "allow": True,   # True activates computation on GPU (if GPU is available)
        "mixed_precision": False,  # true activates mixed precision training
        "multi_gpu": False,  # true activates multi-GPU parallelization approaches
    }
}