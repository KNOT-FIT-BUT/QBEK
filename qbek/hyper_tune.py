# -*- coding: UTF-8 -*-
""""
Created on 20.04.21
Module for hyperparameters tuning.

:author:     Martin Dočekal
"""
import copy
import logging
import math
import os
import sys
import traceback
from pathlib import Path
from typing import Callable, Dict, Any, Tuple

from hyperopt import STATUS_FAIL, STATUS_OK


class FitWrapperForHyperTune:
    """
    Wraps training procedure with its configuration for purposes of hyper params tuning.
    """

    def __init__(self, train_procedure: Callable[["TrainConfig"], float], default_config: "TrainConfig"):
        """
        Initialization of wrapper.

        :param train_procedure: Training procedure accepting configuration as first parameter.
        :param default_config: Default configuration for training procedure.
            It is used as default template that will be filled with actual parameters that should be tried.
        """

        self.train_procedure = train_procedure
        self.default_config = default_config

    def __call__(self, trial_config: Dict[str, Any]) -> Dict[str, float]:
        """
        Run single trial of hyper param search.

        :param trial_config: Actual trial configuration.
        :return: [loss] of trial
        """
        try:
            config = copy.deepcopy(self.default_config)

            config["training"]["batch"] = trial_config["batch"]
            config["training"]["max_steps"] = trial_config["max_steps"]
            config["training"]["optimizer"]["learning_rate"] = trial_config["lr"]
            config["training"]["scheduler"]["type"] = trial_config["scheduler_type"]
            config["training"]["scheduler"]["scheduler_warmup_proportion"] = trial_config["scheduler_warmup_proportion"]

            loss = self.train_procedure(config)
            # remove checkpoints
            for f in Path(config["outputs"]["checkpoints"]["dir"]).glob('LAST_HYPER_CHECKPOINT*'):
                os.remove(f)
            return {"loss": loss}
        except Exception as e:
            print(e, flush=True, file=sys.stderr)
            raise e


class FilterWrapperForHyperTune:
    """
    Wraps filter searching procedure with its configuration for purposes of hyper params tuning.
    """

    def __init__(self, filter_procedure: Callable[[Tuple[int, float, int]], float]):
        """
        Initialization of wrapper.

        :param filter_procedure: Filter searching procedure accepting configuration as first parameter.
        """

        self.filter_procedure = filter_procedure

    def __call__(self, trial_config: Dict[str, Any]) -> Dict[str, float]:
        """
        Run single trial of hyper param search.

        :param trial_config: Actual trial configuration.
        :return: [loss] of trial
        """
        try:
            #(max_span_size, score_threshold, n_best)
            return {
                "loss": - self.filter_procedure(
                    (trial_config["max_span_size"], trial_config["score_threshold"], trial_config["n_best"])
                ),
                "status": STATUS_OK,
                "config": trial_config
            }
        except Exception as e:
            logging.error(e)
            logging.error(traceback.format_exc())
            return {'loss': math.inf, 'status': STATUS_FAIL}
