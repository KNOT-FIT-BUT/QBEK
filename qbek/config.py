# -*- coding: UTF-8 -*-
""""
Created on 03.03.21

Module containing configuration.

:author:     Martin Dočekal
"""
import datetime
import os
import re
import socket
from typing import Dict, Any

from pytorch_lightning.callbacks import EarlyStopping, ModelCheckpoint
from windpyutils.config import Config

from qbek.entities import Language


class BaseConfig(Config):
    """
    Enriches default config class with addition features.
    """

    def _validate_transformers(self, config: Dict):
        """
        Validates the loaded configuration transformers part.

        :param config: Loaded configuration. May be changed in place in this method.
        :raise ValueError: Invalid value for a parameter or missing parameter.
        """

        if "tokenizer" not in config or not isinstance(config["tokenizer"], str):
            raise ValueError("[transformers][tokenizer]You must provide tokenizer.")

        if "cache" not in config or (config["cache"] is not None and not isinstance(config["cache"], str)):
            raise ValueError("[transformers] You must provide cache.")

        if config["cache"] is not None:
            config["cache"] = self.translate_file_path(config["cache"])

        if "tokenizer" not in config or not isinstance(config["tokenizer"], str):
            raise ValueError("[transformers][tokenizer]You must provide tokenizer.")

        if "local_files_only" not in config or not isinstance(config["local_files_only"], bool):
            raise ValueError("[transformers][local_files_only]You must local_files_only boolean.")


class TrainConfig(BaseConfig):
    """
    Config structure for KeywordsExtractor training.
    """

    def validate(self, config: Dict):
        """
        Validates the loaded configuration.

        :param config: Loaded configuration. May be changed in place in this method.
        :raise ValueError: Invalid value for a parameter or missing parameter.
        """
        if "name" not in config or not isinstance(config["name"], str):
            raise ValueError("Missing experiment [name].")

        if "fixed_seed" not in config or \
                (config["fixed_seed"] is not None and not isinstance(config["fixed_seed"], int)):
            raise ValueError("The [fixed_seed] should be None or int.")

        # add stamp to config
        timestamp = datetime.datetime.now().strftime("%Y_%m_%d_%H_%M_%S")
        experiment_name = re.sub(r"\s", '_', config["name"])
        config["name_without_spaces"] = experiment_name
        config["stamp_without_exp_name"] = timestamp + "_" + socket.gethostname()
        config["stamp"] = experiment_name + "_" + config["stamp_without_exp_name"]

        for section in ["model", "doc_emb", "transformers", "outputs", "validation", "training", "filters", "resume",
                        "gpu"]:
            if section not in config:
                ValueError(f"The [{section}] section is missing.")

        self._validate_model(config["model"])
        self._validate_transformers(config["transformers"])
        self._validate_outputs(config["outputs"])
        self._validate_training(config["training"])
        self._validate_validation(config["validation"])
        self._validate_resume(config["resume"])
        self._validate_gpu(config["gpu"])

    def config_2_trainer_parameters(self) -> Dict[str, Any]:
        """
        Converts config to trainer parameters.

        :return: dict of parameters for trainer
        """

        res = {
            "gpus": (-1 if self["gpu"]["multi_gpu"] else 1) if self["gpu"]["allow"] else None,
            "accumulate_grad_batches": self["training"]["accu_grad"],
            "max_epochs": self["training"]["max_epochs"],
            "max_steps": self["training"]["max_steps"],
            "callbacks": [ModelCheckpoint(
                monitor='val_loss',
                dirpath=self["outputs"]["checkpoints"]["dir"],
                filename=self["name_without_spaces"] + "_" + self["stamp_without_exp_name"] + \
                         "_{epoch}_{step}_{val_loss:.6f}",
                save_top_k=self["outputs"]["checkpoints"]["save_top_k"],
                mode='min'
            )]
        }
        if self["validation"]["steps"] is not None:
            res["val_check_interval"] = self["validation"]["steps"] * self["training"]["accu_grad"]

        if self["training"]["early_stopping"] is not None:
            res["callbacks"].append(EarlyStopping(
                monitor='val_loss',
                patience=self["training"]["early_stopping"]
            ))

        if res["gpus"] == -1:
            res["accelerator"] = "ddp"

        if self["gpu"]["allow"] and self["gpu"]["mixed_precision"]:
            res["precision"] = 16

        if self["training"]["max_grad_norm"] is not None:
            res["gradient_clip_val"] = self["training"]["max_grad_norm"]

        if self["resume"]["activate"] and not self["resume"]["just_model"]:
            res["resume_from_checkpoint"] = self["resume"]["checkpoint"]

        return res

    def _validate_model(self, config: Dict):
        """
        Validates the loaded configuration transformers part.

        :param config: Loaded configuration. May be changed in place in this method.
        :raise ValueError: Invalid value for a parameter or missing parameter.
        """

        if "type" not in config or config["type"] not in {"independent"}:
            raise ValueError("You must provide valid [model][type]. (independent)")

        if "transformer" not in config or not isinstance(config["transformer"], str):
            raise ValueError("You must provide valid [model][transformer].")

        if "init_weights" not in config or not isinstance(config["init_weights"], bool):
            raise ValueError("You must provide valid [model][init_weights].")

        if "freeze_transformer" not in config or not isinstance(config["freeze_transformer"], bool):
            raise ValueError("You must provide valid [model][freeze_transformer].")

    def _validate_outputs(self, config: Dict):
        """
        Validates the loaded configuration outputs part.

        :param config: Loaded configuration. May be changed in place in this method.
        :raise ValueError: Invalid value for a parameter or missing parameter.
        """
        if "checkpoints" not in config:
            raise ValueError("The [outputs][checkpoints] section is missing.")

        config_checkpoints = config["checkpoints"]

        if "dir" not in config_checkpoints or not isinstance(config_checkpoints["dir"], str):
            raise ValueError("[outputs][checkpoints] You must dir (directory where the checkpoints will be saved).")

        config_checkpoints["dir"] = self.translate_file_path(config_checkpoints["dir"])

        if "save_top_k" not in config_checkpoints or not isinstance(config_checkpoints["save_top_k"], int):
            raise ValueError("[outputs][checkpoints] "
                             "You must provide save_top_k that will be integer.")

        if "results" not in config or (not isinstance(config["results"], str) and config["results"] is not None):
            raise ValueError("[outputs] You must provide results that will be path to folder or None.")

        if config["results"] is not None:
            config["results"] = self.translate_file_path(config["results"])

        if "logs" not in config or not isinstance(config["logs"], str):
            raise ValueError("You must provide [outputs][logs].")

        config["logs"] = self.translate_file_path(config["logs"])

    def _validate_training(self, config: Dict):
        """
        Validates the loaded configuration training part.

        :param config: Loaded configuration. May be changed in place in this method.
        :raise ValueError: Invalid value for a parameter or missing parameter.
        """

        if "dataset" not in config:
            ValueError(f"The [training][dataset] section is missing.")

        dataset_config = config["dataset"]
        if "path" not in dataset_config or not isinstance(dataset_config["path"], str):
            raise ValueError("You must provide path to training dataset. ([training][dataset][path])")

        dataset_config["path"] = self.translate_file_path(dataset_config["path"])

        if not os.path.exists(dataset_config["path"]):
            raise ValueError("The training dataset does not exists. ([training][dataset][path])")

        if "workers" not in dataset_config or not isinstance(dataset_config["workers"], int) \
                or dataset_config["workers"] < 0:
            raise ValueError("You must provide [training][dataset][workers] that will be non-negative integer.")

        if "prep_workers" not in dataset_config or not isinstance(dataset_config["prep_workers"], int) \
                or dataset_config["prep_workers"] < 0:
            raise ValueError("You must provide [training][dataset][prep_workers] that will be non-negative integer.")

        if "add_title" not in dataset_config or not isinstance(dataset_config["add_title"], bool) \
                or dataset_config["add_title"] < 0:
            raise ValueError("You must provide [training][dataset][add_title] that will be boolean.")

        if "batch" not in config or not isinstance(config["batch"], int) or config["batch"] <= 0:
            raise ValueError("You must provide [training][batch] that will be positive integer.")

        if "max_grad_norm" not in config or \
                not (config["max_grad_norm"] is None or isinstance(config["max_grad_norm"], float)) or \
                (isinstance(config["max_grad_norm"], float) and config["max_grad_norm"] <= 0):
            raise ValueError("You must provide [training][max_grad_norm] that will be positive number or None.")

        if "accu_grad" not in config or not isinstance(config["accu_grad"], int) or config["accu_grad"] < 1:
            raise ValueError("You must provide [training][accu_grad] that will be positive integer.")

        if "max_epochs" not in config or not isinstance(config["max_epochs"], int) or config["max_epochs"] <= 0:
            raise ValueError("You must provide [training][max_epochs] that will be positive integer.")

        if "max_steps" not in config or not isinstance(config["max_steps"], int) or config["max_steps"] <= 0:
            raise ValueError("You must provide [training][max_steps] that will be positive integer.")

        if "early_stopping" not in config or \
                not (config["early_stopping"] is None or isinstance(config["early_stopping"], int)) or \
                (isinstance(config["early_stopping"], int) and config["early_stopping"] <= 0):
            raise ValueError("You must provide [training][early_stopping] that will be positive positive integer "
                             "or None.")

        if "optimizer" not in config:
            raise ValueError("The [training][optimizer] section is missing.")

        config_optimizer = config["optimizer"]

        if "learning_rate" not in config_optimizer or not isinstance(config_optimizer["learning_rate"], float):
            raise ValueError("You must provide valid [training][optimizer][learning_rate].")

        if "weight_decay" not in config_optimizer or not isinstance(config_optimizer["weight_decay"], float):
            raise ValueError("You must provide valid [training][optimizer][weight_decay].")

        config_scheduler = config["scheduler"]

        if "type" not in config_scheduler or config_scheduler["type"] not in {None, "None", "linear", "cosine",
                                                                              "constant"}:
            raise ValueError("You must provide valid [training][scheduler][type]. (None, linear, cosine, constant)")

        if config_scheduler["type"] == "None":
            config_scheduler["type"] = None

        if "scheduler_warmup_proportion" not in config_scheduler or \
                not isinstance(config_scheduler["scheduler_warmup_proportion"], float) or \
                config_scheduler["scheduler_warmup_proportion"] < 0:
            raise ValueError("You must provide [training][scheduler][scheduler_warmup_proportion] that will be "
                             "non-negative number.")

        if "online_weighting" not in config or not isinstance(config["online_weighting"], bool) \
                or config["online_weighting"] < 0:
            raise ValueError("You must provide [training][online_weighting] that will be boolean.")

    def _validate_validation(self, config: Dict):
        """
        Validates the loaded configuration validation part.

        :param config: Loaded configuration. May be changed in place in this method.
        :raise ValueError: Invalid value for a parameter or missing parameter.
        """
        if "dataset" not in config:
            ValueError(f"The [training][dataset] section is missing.")

        dataset_config = config["dataset"]

        if "path" not in dataset_config or not isinstance(dataset_config["path"], str):
            raise ValueError("You must provide validation dataset. ([validation][dataset][path])")

        dataset_config["path"] = self.translate_file_path(dataset_config["path"])

        if not os.path.exists(dataset_config["path"]):
            raise ValueError("The validation dataset does not exists. ([validation][dataset][path])")

        if "workers" not in dataset_config or not isinstance(dataset_config["workers"], int) \
                or dataset_config["workers"] < 0:
            raise ValueError("You must provide [validation][dataset][workers] that will be non-negative integer.")

        if "prep_workers" not in dataset_config or not isinstance(dataset_config["prep_workers"], int) \
                or dataset_config["prep_workers"] < 0:
            raise ValueError("You must provide [validation][dataset][prep_workers] that will be non-negative integer.")

        if "add_title" not in dataset_config or not isinstance(dataset_config["add_title"], bool) \
                or dataset_config["add_title"] < 0:
            raise ValueError("You must provide [validation][dataset][add_title] that will be boolean.")

        if "steps" not in config or not (config["steps"] is None or isinstance(config["steps"], int)) or \
                (isinstance(config["steps"], int) and config["steps"] <= 0):
            raise ValueError("You must provide [validation][steps] that will be positive integer or None.")

        if "batch" not in config or not isinstance(config["batch"], int) or config["batch"] <= 0:
            raise ValueError("You must provide [validation][batch] that will be positive integer.")

    def _validate_resume(self, config: Dict):
        """
        Validates the loaded configuration resume part.

        :param config: Loaded configuration. May be changed in place in this method.
        :raise ValueError: Invalid value for a parameter or missing parameter.
        """
        if "activate" not in config or not isinstance(config["activate"], bool):
            raise ValueError("You must provide [resume][activate] boolean.")

        if config["activate"]:
            if "checkpoint" not in config or not isinstance(config["checkpoint"], str):
                raise ValueError("You must provide [resume][checkpoint].")

            config["checkpoint"] = self.translate_file_path(config["checkpoint"])

            if not os.path.exists(config["checkpoint"]):
                raise ValueError("The [resume][checkpoint] does not exists.")

            if "just_model" not in config or not isinstance(config["just_model"], bool):
                raise ValueError("You must provide [resume][just_model] boolean.")

    def _validate_gpu(self, config: Dict):
        """
        Validates the loaded configuration gpu part.

        :param config: Loaded configuration. May be changed in place in this method.
        :raise ValueError: Invalid value for a parameter or missing parameter.
        """

        if "allow" not in config or not isinstance(config["allow"], bool):
            raise ValueError("You must provide [gpu][allow] boolean.")

        if "mixed_precision" not in config or not isinstance(config["mixed_precision"], bool):
            raise ValueError("You must provide [gpu][mixed_precision] boolean.")

        if "multi_gpu" not in config or not isinstance(config["multi_gpu"], bool):
            raise ValueError("You must provide [gpu][multi_gpu] boolean.")


class HyperConfig(TrainConfig):
    """
    Config structure for KeywordsExtractor hyper params search.
    """

    def config_2_trainer_parameters(self) -> Dict[str, Any]:
        """
        Converts config to trainer parameters.

        :return: dict of parameters for trainer
        """
        params = super().config_2_trainer_parameters()
        for c in params["callbacks"]:
            if isinstance(c, ModelCheckpoint):
                c.filename = "LAST_HYPER_CHECKPOINT"
        return params

    def validate(self, config: Dict):
        """
        Validates the loaded configuration.

        :param config: Loaded configuration. May be changed in place in this method.
        :raise ValueError: Invalid value for a parameter or missing parameter.
        """
        super().validate(config)

        for section in ["space"]:
            if section not in config:
                ValueError(f"The [{section}] section is missing.")

        self._validate_space(config["space"])

    def _validate_space(self, config: Dict):
        """
        Validates the loaded configuration space part.

        :param config: Loaded configuration. May be changed in place in this method.
        :raise ValueError: Invalid value for a parameter or missing parameter.
        """
        for section in ["batch", "max_steps", "lr", "scheduler", "scheduler_warmup_proportion", "samples"]:
            if section not in config:
                ValueError(f"The [space][{section}] section is missing.")

        if "samples" not in config or not isinstance(config["samples"], int) or config["samples"] <= 0:
            raise ValueError("You must provide [space][samples] that will be positive integer.")


class FiltersConfig(BaseConfig):
    """
    Config structure for enriching trained KeywordsExtractor with filters.
    """

    def validate(self, config: Dict):
        """
        Validates the loaded configuration.

        :param config: Loaded configuration. May be changed in place in this method.
        :raise ValueError: Invalid value for a parameter or missing parameter.
        """

        if "result" not in config or not isinstance(config["result"], str):
            raise ValueError("Missing valid [result].")

        config["result"] = self.translate_file_path(config["result"])

        if "language" not in config or not isinstance(config["language"], str):
            raise ValueError("Missing valid [language].")

        try:
            config["language"] = Language(config["language"])
        except ValueError:
            raise ValueError("Invalid [language].")

        if "trials" not in config or not isinstance(config["trials"], int) or config["trials"] <=0:
            raise ValueError("The [trials] must be positive integer.")

        for section in ["model", "doc_emb", "validation", "transformers", "filters", "gpu"]:
            if section not in config:
                ValueError(f"The [{section}] section is missing.")

        self._validate_model(config["model"])
        self._validate_validation(config["validation"])
        self._validate_transformers(config["transformers"])
        self._validate_filters(config["filters"])
        self._validate_gpu(config["gpu"])

    def _validate_model(self, config: Dict):
        """
        Validates the loaded configuration transformers part.

        :param config: Loaded configuration. May be changed in place in this method.
        :raise ValueError: Invalid value for a parameter or missing parameter.
        """

        if "type" not in config or config["type"] not in {"independent", "conditional"}:
            raise ValueError("You must provide valid [model][type]. (independent, conditional)")

        if "checkpoint" not in config or not isinstance(config["checkpoint"], str):
            raise ValueError("You must provide valid [model][checkpoint].")

        config["checkpoint"] = self.translate_file_path(config["checkpoint"])

        if not os.path.exists(config["checkpoint"]):
            raise ValueError("The [model][checkpoint] does not exists.")

    def _validate_validation(self, config: Dict):
        """
        Validates the loaded configuration validation part.

        :param config: Loaded configuration. May be changed in place in this method.
        :raise ValueError: Invalid value for a parameter or missing parameter.
        """
        if "dataset" not in config:
            ValueError(f"The [validation][dataset] section is missing.")

        dataset_config = config["dataset"]

        if "path" not in dataset_config or not isinstance(dataset_config["path"], str):
            raise ValueError("You must provide validation dataset. ([validation][dataset][path])")

        dataset_config["path"] = self.translate_file_path(dataset_config["path"])

        if not os.path.exists(dataset_config["path"]):
            raise ValueError("The validation dataset does not exists. ([validation][dataset][path])")

        if "workers" not in dataset_config or not isinstance(dataset_config["workers"], int) \
                or dataset_config["workers"] < 0:
            raise ValueError("You must provide [validation][dataset][workers] that will be non-negative integer.")

        if "add_title" not in dataset_config or not isinstance(dataset_config["add_title"], bool):
            raise ValueError("You must provide [validation][dataset][add_title] that will be boolean.")

        if "batch" not in config or not isinstance(config["batch"], int) or config["batch"] <= 0:
            raise ValueError("You must provide [validation][batch] that will be positive integer.")

    def _validate_filters(self, config: Dict):
        """
        Validates the loaded configuration filters part.

        :param config: Loaded configuration. May be changed in place in this method.
        :raise ValueError: Invalid value for a parameter or missing parameter.
        """
        if "max_span_size" not in config or \
                not (isinstance(config["max_span_size"], int) or config["max_span_size"] == "auto") or \
                (isinstance(config["max_span_size"], int) and config["max_span_size"] <= 0):
            raise ValueError("You must provide [filters][max_span_size] that will be positive integer or auto.")

        if config["max_span_size"] == "auto":
            config["max_span_size"] = None

        if "n_best" not in config or \
                not (isinstance(config["n_best"], int) or config["n_best"] == "auto") or \
                (isinstance(config["n_best"], int) and config["n_best"] <= 0):
            raise ValueError("You must provide [filters][n_best] that will be positive integer or auto.")

        if config["n_best"] == "auto":
            config["n_best"] = None

        if "score_threshold" not in config or \
                not (isinstance(config["score_threshold"], float) or config["score_threshold"] == "auto"):
            raise ValueError("You must provide valid [filters][score_threshold].")

        if config["score_threshold"] == "auto":
            config["score_threshold"] = None

    def _validate_gpu(self, config: Dict):
        """
        Validates the loaded configuration gpu part.

        :param config: Loaded configuration. May be changed in place in this method.
        :raise ValueError: Invalid value for a parameter or missing parameter.
        """

        if "allow" not in config or not isinstance(config["allow"], bool):
            raise ValueError("You must provide [gpu][allow] boolean.")

        if "mixed_precision" not in config or not isinstance(config["mixed_precision"], bool):
            raise ValueError("You must provide [gpu][mixed_precision] boolean.")


class EvalConfig(BaseConfig):
    """
    Config structure for KeywordsExtractor evaluation.
    """

    def validate(self, config: Dict):
        """
        Validates the loaded configuration.

        :param config: Loaded configuration. May be changed in place in this method.
        :raise ValueError: Invalid value for a parameter or missing parameter.
        """

        if "language" not in config or not isinstance(config["language"], str):
            raise ValueError("Missing valid [language].")

        try:
            config["language"] = Language(config["language"])
        except ValueError:
            raise ValueError("Invalid [language].")

        for section in ["metrics", "doc_emb", "model", "outputs", "validation", "transformers", "gpu"]:
            if section not in config:
                ValueError(f"The [{section}] section is missing.")

        if len(config["metrics"]) == 0:
            ValueError(f"There must be at least one metric in [metrics].")

        for m in config["metrics"]:
            if not isinstance(m, str) or m not in {"multilabel_identification"}:
                ValueError(f"Invalid metric {m} in [metrics].")

        self._validate_model(config["model"])
        self._validate_validation(config["validation"])
        self._validate_outputs(config["outputs"])
        self._validate_transformers(config["transformers"])
        self._validate_gpu(config["gpu"])

    def _validate_model(self, config: Dict):
        """
        Validates the loaded configuration transformers part.

        :param config: Loaded configuration. May be changed in place in this method.
        :raise ValueError: Invalid value for a parameter or missing parameter.
        """

        if "type" not in config or config["type"] not in {"independent", "conditional"}:
            raise ValueError("You must provide valid [model][type]. (independent, conditional)")

        if "checkpoint" not in config or not isinstance(config["checkpoint"], str):
            raise ValueError("You must provide valid [model][checkpoint].")

        config["checkpoint"] = self.translate_file_path(config["checkpoint"])

        if not os.path.exists(config["checkpoint"]):
            raise ValueError("The [model][checkpoint] does not exists.")

    def _validate_outputs(self, config: Dict):
        """
        Validates the loaded configuration outputs part.

        :param config: Loaded configuration. May be changed in place in this method.
        :raise ValueError: Invalid value for a parameter or missing parameter.
        """
        if "predictions" not in config or not isinstance(config["predictions"], str):
            raise ValueError("Missing valid [predictions].")

        config["predictions"] = self.translate_file_path(config["predictions"])

        if "results" not in config or not isinstance(config["results"], str):
            raise ValueError("Missing valid [results].")

        config["results"] = self.translate_file_path(config["results"])

    def _validate_validation(self, config: Dict):
        """
        Validates the loaded configuration validation part.

        :param config: Loaded configuration. May be changed in place in this method.
        :raise ValueError: Invalid value for a parameter or missing parameter.
        """
        if "dataset" not in config:
            ValueError(f"The [training][dataset] section is missing.")

        dataset_config = config["dataset"]

        if "path" not in dataset_config or not isinstance(dataset_config["path"], str):
            raise ValueError("You must provide validation dataset. ([validation][dataset][path])")

        dataset_config["path"] = self.translate_file_path(dataset_config["path"])

        if not os.path.exists(dataset_config["path"]):
            raise ValueError("The validation dataset does not exists. ([validation][dataset][path])")

        if "workers" not in dataset_config or not isinstance(dataset_config["workers"], int) \
                or dataset_config["workers"] < 0:
            raise ValueError("You must provide [validation][dataset][workers] that will be non-negative integer.")

        if "add_title" not in dataset_config or not isinstance(dataset_config["add_title"], bool):
            raise ValueError("You must provide [validation][dataset][add_title] that will be boolean.")

        if "batch" not in config or not isinstance(config["batch"], int) or config["batch"] <= 0:
            raise ValueError("You must provide [validation][batch] that will be positive integer.")

    def _validate_gpu(self, config: Dict):
        """
        Validates the loaded configuration gpu part.

        :param config: Loaded configuration. May be changed in place in this method.
        :raise ValueError: Invalid value for a parameter or missing parameter.
        """

        if "allow" not in config or not isinstance(config["allow"], bool):
            raise ValueError("You must provide [gpu][allow] boolean.")

        if "mixed_precision" not in config or not isinstance(config["mixed_precision"], bool):
            raise ValueError("You must provide [gpu][mixed_precision] boolean.")


class UseConfig(BaseConfig):
    """
    Config structure for KeywordsExtractor using (extracting/identifying).
    """

    def validate(self, config: Dict):
        """
        Validates the loaded configuration.

        :param config: Loaded configuration. May be changed in place in this method.
        :raise ValueError: Invalid value for a parameter or missing parameter.
        """

        for section in ["model", "doc_emb", "testing", "transformers", "gpu"]:
            if section not in config:
                ValueError(f"The [{section}] section is missing.")

        if "results" not in config or not isinstance(config["results"], str):
            raise ValueError("Missing valid [results].")

        config["results"] = self.translate_file_path(config["results"])

        self._validate_model(config["model"])
        self._validate_testing(config["testing"])
        self._validate_transformers(config["transformers"])
        self._validate_gpu(config["gpu"])

    def _validate_model(self, config: Dict):
        """
        Validates the loaded configuration transformers part.

        :param config: Loaded configuration. May be changed in place in this method.
        :raise ValueError: Invalid value for a parameter or missing parameter.
        """

        if "type" not in config or config["type"] not in {"independent", "conditional"}:
            raise ValueError("You must provide valid [model][type]. (independent, conditional)")

        if "checkpoint" not in config or not isinstance(config["checkpoint"], str):
            raise ValueError("You must provide valid [model][checkpoint].")

        config["checkpoint"] = self.translate_file_path(config["checkpoint"])

        if not os.path.exists(config["checkpoint"]):
            raise ValueError("The [model][checkpoint] does not exists.")

    def _validate_testing(self, config: Dict):
        """
        Testing the loaded configuration validation part.

        :param config: Loaded configuration. May be changed in place in this method.
        :raise ValueError: Invalid value for a parameter or missing parameter.
        """
        if "dataset" not in config:
            ValueError(f"The [dataset] section is missing.")

        dataset_config = config["dataset"]

        if "path" in dataset_config and dataset_config["path"] is not None:
            # voluntary
            dataset_config["path"] = self.translate_file_path(dataset_config["path"])

            if not os.path.exists(dataset_config["path"]):
                raise ValueError("The dataset does not exists. ([dataset][path])")

        if "workers" not in dataset_config or not isinstance(dataset_config["workers"], int) \
                or dataset_config["workers"] < 0:
            raise ValueError("You must provide [dataset][workers] that will be non-negative integer.")

        if "add_title" not in dataset_config or not isinstance(dataset_config["add_title"], bool) \
                or dataset_config["add_title"] < 0:
            raise ValueError("You must provide [dataset][add_title] that will be boolean.")

        if "batch" not in config or not isinstance(config["batch"], int) or config["batch"] <= 0:
            raise ValueError("You must provide [batch] that will be positive integer.")

    def _validate_gpu(self, config: Dict):
        """
        Validates the loaded configuration gpu part.

        :param config: Loaded configuration. May be changed in place in this method.
        :raise ValueError: Invalid value for a parameter or missing parameter.
        """

        if "allow" not in config or not isinstance(config["allow"], bool):
            raise ValueError("You must provide [gpu][allow] boolean.")

        if "mixed_precision" not in config or not isinstance(config["mixed_precision"], bool):
            raise ValueError("You must provide [gpu][mixed_precision] boolean.")
