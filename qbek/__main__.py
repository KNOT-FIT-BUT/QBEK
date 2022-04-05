# -*- coding: UTF-8 -*-
""""
Created on 07.10.19
KeywordsExtractor
Extraction of keywords from a text.

:author:     Martin Dočekal
"""
import argparse
import atexit
import datetime
import logging
import multiprocessing
import os
import pprint
import random
import socket
import sys
from argparse import RawTextHelpFormatter
from multiprocessing import active_children
from pathlib import Path
from typing import Optional, Dict, Any, List, Tuple, Union, Generator, Set

import pandas as pd
import torch
import transformers
from pytorch_lightning import Trainer, seed_everything
from pytorch_lightning.callbacks import ModelCheckpoint
from ray import tune
from torch.optim import AdamW
from torch.utils.data import DataLoader
from tqdm import tqdm
from windpyutils.args import ExceptionsArgumentParser, ArgumentParserError

from qbek.auto_filter import AutoFilter
from qbek.batch import Batch
from qbek.config import TrainConfig, UseConfig, FiltersConfig, EvalConfig, HyperConfig
from qbek.datasets.czech_library import CzechLibraryDataModule
from qbek.entities import DocumentResults, Document, Language
from qbek.hyper_tune import FitWrapperForHyperTune
from qbek.metrics.extraction_generation import ExtractionGenerationMetric
from qbek.models.independent import Independent
from qbek.models.model import Model
from qbek.results import ResultsSaving
from qbek.tokenizer import AlphaTokenizer
from qbek.training.optimizer_factory import AnyOptimizerFactory
from qbek.training.scheduler_factory import AnySchedulerFactory, SchedulerFactory
from qbek.utils.top_keeper import TopKeeper

os.environ["TOKENIZERS_PARALLELISM"] = "false"


class ArgumentsManager(object):
    """
    Parsers arguments for script.
    """

    @classmethod
    def parse_args(cls):
        """
        Performs arguments parsing.

        :param cls: arguments class
        :returns: Parsed arguments.
        """

        parser = ExceptionsArgumentParser(description="Script for training/using model for keyphrase "
                                                      "identification/extraction.",
                                          formatter_class=RawTextHelpFormatter)

        subparsers = parser.add_subparsers()

        train_parser = subparsers.add_parser('train', help="Tool for keywords extractor training.")
        train_parser.add_argument("config", help="Path to file containing the configuration.", type=str)
        train_parser.set_defaults(func=call_train)

        hyper_parser = subparsers.add_parser('hyper', help="Hyperparameters search.")
        hyper_parser.add_argument("config", help="Path to file containing the configuration.", type=str)
        hyper_parser.set_defaults(func=call_hyper)

        filters_parser = subparsers.add_parser('filters', help="Tool for searching parameters for filtration.")
        filters_parser.add_argument("config", help="Path to file containing the configuration.", type=str)
        filters_parser.set_defaults(func=call_filters)

        eval_parser = subparsers.add_parser('eval', help="Tool for evaluation of model.")
        eval_parser.add_argument("config", help="Path to file containing the configuration.", type=str)
        eval_parser.set_defaults(func=call_eval)

        extract_parser = subparsers.add_parser('extract',
                                               help="Extracts keywords in text. Result is printed to stdout")
        extract_parser.add_argument("config", help="Path to file containing the configuration.", type=str)
        extract_parser.set_defaults(func=call_extract)

        subparsers_for_help = {
            'train': train_parser,
            'hyper': hyper_parser,
            'filters': filters_parser,
            'eval': eval_parser,
            'extract': extract_parser,
        }

        if len(sys.argv) < 2:
            parser.print_help()
            return None
        try:
            parsed = parser.parse_args()

        except ArgumentParserError as e:
            for name, subParser in subparsers_for_help.items():
                if name == sys.argv[1]:
                    subParser.print_help()
                    break
            print("\n" + str(e), file=sys.stdout, flush=True)
            return None

        return parsed


def init_scheduler_factory(config: Dict[str, Any]) -> Optional[SchedulerFactory]:
    """
    Initialization of lr scheduler factory.

    :param config: experiment's configuration
    :return: Created scheduler factory.
    :rtype: Optional[SchedulerFactory]
    """

    num_of_warm_up_steps = config["training"]["scheduler"]["scheduler_warmup_proportion"] * \
                           config["training"]["max_steps"]

    if config["training"]["scheduler"]["type"] == "linear":
        factory = AnySchedulerFactory(creator=transformers.get_linear_schedule_with_warmup,
                                      attr={
                                          "num_warmup_steps": num_of_warm_up_steps,
                                          "num_training_steps": config["training"]["max_steps"]
                                      })
    elif config["training"]["scheduler"]["type"] == "cosine":
        factory = AnySchedulerFactory(creator=transformers.get_cosine_schedule_with_warmup,
                                      attr={
                                          "num_warmup_steps": num_of_warm_up_steps,
                                          "num_training_steps": config["training"]["max_steps"],
                                          "num_cycles": 0.5
                                      })
    elif config["training"]["scheduler"]["type"] == "constant":
        factory = AnySchedulerFactory(creator=transformers.get_constant_schedule_with_warmup,
                                      attr={
                                          "num_warmup_steps": num_of_warm_up_steps,
                                      })
    else:
        factory = None

    return factory


def evaluate_pred(eval_res_path: str, metric: ExtractionGenerationMetric) -> Any:
    """
    Evaluates prediction in given file.

    :param eval_res_path: Path to file with predictions.
    :param metric: metrics that will be used for evaluation
    :return: evaluation results
    """

    results = pd.read_csv(eval_res_path, delimiter="\t")

    predictions = [
        [] if pd.isna(doc_pred) else doc_pred.split(ResultsSaving.KEYPHRASES_SEPARATOR)
        for doc_pred in results["predicted"].values
    ]
    ground_truths = [
        [] if pd.isna(doc_pred) else doc_pred.split(ResultsSaving.KEYPHRASES_SEPARATOR)
        for doc_pred in results["extractive ground truth"].values
    ]

    return metric.eval(predictions, ground_truths)


def get_ground_truths(path_to: str, language: Language) -> List[Set[str]]:
    """
    Gets ground truths from dataset on given path.

    :param path_to: path to the dataset
    :param language: Language of dataset. Is used for lemmatization normalization.
    :return: ground truths
    """
    Document.shared_lemmatizer = language.lemmatizer
    ground_truths = []
    with open(path_to, "r") as f:
        for line in f:
            ground_truths.append(Document.from_json(line).annotated_ext_kp)  # interested in extractive only

    return ground_truths


@torch.no_grad()
def predict(model: Model, data_loader: DataLoader, mixed_precision: bool, verbose: bool = True, allow_gpu: bool = True) \
        -> Generator[Union[
                         List[  # list of batches for a document
                             Tuple[torch.Tensor, Batch],  # when: model.pred_span_universe_matrix is True
                         ],
                         DocumentResults
                     ],
                     None, None]:
    """
    Makes predictions for validation dataset

    :param model: This model will be used for predictions.
    :param data_loader: loader of dataset
    :param mixed_precision: true enables mixed precision
    :param verbose: Whether to show or not a progress bar.
    :param allow_gpu: Can the GPU could be used.
    :return: generates span universe matrices for a documents and batches itself if
            model.pred_span_universe_matrix is True
            else generates
                List of DocumentResults
    """
    trainer_config = {}
    if allow_gpu:
        trainer_config["gpus"] = 1
    if mixed_precision:
        trainer_config["precision"] = 16
    if not verbose:
        trainer_config["progress_bar_refresh_rate"] = 1

    trainer = Trainer(**trainer_config)
    model.eval()
    act_document_res = []

    def tak_first_n(document_res: List[List[Tuple[List[str], List[float], int]]]) -> DocumentResults:
        spans = TopKeeper(model.n_best)
        doc_line_offset = None
        for batch_res in document_res:
            for doc_sample in batch_res:
                if doc_line_offset is None:
                    doc_line_offset = doc_sample[2]

                for span, score in zip(doc_sample[0], doc_sample[1]):
                    spans.push(score, span)

        final_spans = []
        final_scores = []

        for i in range(len(spans)):
            k, s = spans.get_element(i)
            final_spans.append(k)
            final_scores.append(s)

        return DocumentResults(
            documents_line_offset=doc_line_offset,
            spans=final_spans, scores=final_scores
        )

    last_line_offset = None

    for res in tqdm(trainer.predict(model, data_loader), desc="Sorting samples into documents", disable=not verbose):
        # Union[Tuple[torch.Tensor, Batch], List[Tuple[List[str], List[float], int]]]
        if model.pred_span_universe_matrix:
            line_offsets = res[1].line_offsets
        else:
            line_offsets = [x[2] for x in res]

        if last_line_offset is None:
            last_line_offset = line_offsets[0]

        res_split_offset = -1
        for line_offset in line_offsets:
            res_split_offset += 1
            if last_line_offset != line_offset:
                # this sample is from another document
                last_line_offset = line_offset
                if res_split_offset > 0:
                    # not a sample on start, we should split current batch

                    if model.pred_span_universe_matrix:
                        batch_one, batch_two = res[1].split(res_split_offset)
                        act_document_res.append((res[0][:res_split_offset], batch_one))
                        res = (res[0][res_split_offset:], batch_two)
                    else:
                        act_document_res.append(res[:res_split_offset])
                        res = res[res_split_offset:]

                    res_split_offset = 0

                if not model.pred_span_universe_matrix:
                    act_document_res = tak_first_n(act_document_res)

                yield act_document_res
                act_document_res = []

        act_document_res.append(res)

    if len(act_document_res) > 0:
        if not model.pred_span_universe_matrix:
            act_document_res = tak_first_n(act_document_res)
        yield act_document_res


def train(config: TrainConfig) -> float:
    """
    Starts training with given configuration.

    :param config: training configuration
    :return: best loss
    """
    trainer_params = config.config_2_trainer_parameters()

    if trainer_params["gpus"] == -1:
        # torch lightning will create process for each gpu and all of these process would try to create all threads
        # which would cause significant slowdown, so we distribute them evenly
        torch.set_num_threads(max(1, int(multiprocessing.cpu_count() / torch.cuda.device_count())))

    logging.info("\n" + pprint.pformat(config))

    if config["fixed_seed"] is None:
        seed = int(random.getrandbits(32))
    else:
        seed = config["fixed_seed"]

    logging.info(f"The seed for this experiment is {seed}.")

    seed_everything(seed, True)

    data_module = CzechLibraryDataModule(config)

    data_module.prepare_data()

    optimizer = AnyOptimizerFactory(AdamW, attr={
        "lr": config["training"]["optimizer"]["learning_rate"],
        "weight_decay": config["training"]["optimizer"]["weight_decay"]
    })

    scheduler = init_scheduler_factory(config)

    init_weights = ("resume" in config and not config["resume"]["activate"]) & config["model"]["init_weights"]

    if "resume" in config and config["resume"]["activate"] and not config["resume"]["just_model"]:
        raise NotImplementedError()

    if "resume" in config and config["resume"]["activate"] and config["resume"]["just_model"]:
        model = Independent.load_from_checkpoint(config["resume"]["checkpoint"],
                                                 cache_dir=config["transformers"]["cache"],
                                                 local_files_only=config["transformers"]["local_files_only"],
                                                 optimizer=optimizer,
                                                 scheduler=scheduler,
                                                 online_weighting=config["training"]["online_weighting"],
                                                 freeze_transformer=config["model"]["freeze_transformer"])
    else:
        model = Independent(config["model"]["transformer"], config["transformers"]["cache"],
                            local_files_only=config["transformers"]["local_files_only"],
                            optimizer=optimizer,
                            scheduler=scheduler,
                            init_pretrained_transformer_weights=init_weights,
                            online_weighting=config["training"]["online_weighting"],
                            freeze_transformer=config["model"]["freeze_transformer"])

    trainer = Trainer(**trainer_params)
    trainer.fit(model, data_module)

    # get best checkpoint path
    logging.info("Training is done.")
    best_model_path = None
    best_loss = None

    for callback in trainer_params["callbacks"]:
        if isinstance(callback, ModelCheckpoint):
            best_model_path = callback.best_model_path
            best_loss = callback.best_model_score.item()
            break

    logging.info(f"Best checkpoint: {Path(best_model_path).name}")
    logging.info(f"Best loss {os.getpid()}: {best_loss}")

    return best_loss


def call_train(args: argparse.Namespace):
    """
    Method for keyphrase extractor training.

    :param args: User arguments.
    """

    config = TrainConfig(args.config)
    setup_logging(config["outputs"]["logs"], config["name_without_spaces"], config["outputs"]["logs"] is not None)
    train(config)


@torch.no_grad()
def call_filters(args: argparse.Namespace):
    """
    Method for keyphrase identification.

    :param args: User arguments.
    """
    config = FiltersConfig(args.config)

    auto_filter = AutoFilter(config["language"].lemmatizer, config["filters"]["n_best"],
                             config["filters"]["max_span_size"], config["filters"]["score_threshold"], config["trials"])

    model = Independent.load_from_checkpoint(config["model"]["checkpoint"],
                                             cache_dir=config["transformers"]["cache"])

    if auto_filter.all_params_known():
        best_combination = (auto_filter.max_span_size, auto_filter.score_threshold, auto_filter.n_best, None)
    else:
        model.pred_span_universe_matrix = True
        data_module = CzechLibraryDataModule(config)
        data_module.setup()

        val_predictions = list(predict(model, data_module.val_dataloader(),
                                       config["gpu"]["mixed_precision"], allow_gpu=config["gpu"]["allow"]))

        best_combination = auto_filter(val_predictions, get_ground_truths(data_module.val.path_to, config["language"]))
        model.pred_span_universe_matrix = False

        print(f"Best found filter is: span size {best_combination[0]}, threshold {best_combination[1]} "
              f"and first n {best_combination[2]}. It reaches F1 {best_combination[3]} on lower case lemma match.")

    model.max_span_size = best_combination[0]
    model.score_threshold = best_combination[1]
    model.n_best = best_combination[2]

    # save the best model
    trainer = Trainer()
    trainer.model = model
    print(f"Saving model with filters to {config['result']}.")
    trainer.save_checkpoint(config["result"], weights_only=True)


@torch.no_grad()
def call_eval(args: argparse.Namespace):
    """
    Method for keyphrase evaluation.

    :param args: User arguments.
    """
    config = EvalConfig(args.config)

    model = Independent.load_from_checkpoint(config["model"]["checkpoint"],
                                             cache_dir=config["transformers"]["cache"])

    print(f'Loaded the {config["model"]["checkpoint"]}.\n'
          f'With filters:\n'
          f'\tn_best: {model.n_best}\n'
          f'\tmax_span_size: {model.max_span_size}\n'
          f'\tscore_threshold: {model.score_threshold}', file=sys.stderr)

    model.pred_span_universe_matrix = False

    data_module = CzechLibraryDataModule(config)
    data_module.setup()

    Document.shared_lemmatizer = config["language"].lemmatizer
    outputs = predict(model, data_module.val_dataloader(), config["gpu"]["mixed_precision"],
                      allow_gpu=config["gpu"]["allow"])

    with open(config["outputs"]["predictions"], "w") as f:
        ResultsSaving.save_results_for_dataset(outputs, data_module.val, f)

    with open(config["outputs"]["results"], "w") as fEval:
        for m in config["metrics"]:
            title = {
                "extractive": "Extractive"
            }[m]

            metric = {
                "extractive": ExtractionGenerationMetric(AlphaTokenizer(), config["language"].lemmatizer)
            }[m]

            print(title, file=fEval)
            metric.print_eval_res(evaluate_pred(config["outputs"]["predictions"], metric), fEval)


@torch.no_grad()
def call_extract(args: argparse.Namespace):
    """
    Method for keyphrase identification.

    :param args: User arguments.
    """
    config = UseConfig(args.config)
    model = Independent.load_from_checkpoint(config["model"]["checkpoint"],
                                             cache_dir=config["transformers"]["cache"])
    model.pred_span_universe_matrix = False

    if "path" not in config["testing"]["dataset"] or config["testing"]["dataset"]["path"] is None:
        raise RuntimeError("Provide path to dataset.")

    data_module = CzechLibraryDataModule(config)
    data_module.setup()

    outputs = predict(model, data_module.test_dataloader(),
                      config["gpu"]["mixed_precision"], allow_gpu=config["gpu"]["allow"])

    with open(config["results"], "w") as f:
        ResultsSaving.save_results_for_dataset(outputs, data_module.test, f)


def call_hyper(args: argparse.Namespace):
    """
    Method for hyper parameters search.

    :param args: User arguments.
    """
    config = HyperConfig(args.config)
    setup_logging(config["outputs"]["logs"], config["name_without_spaces"], config["outputs"]["logs"] is not None)

    wrapper = FitWrapperForHyperTune(train, default_config=config)

    print("RUN START")
    analysis = tune.run(
        wrapper,
        config={
            "batch": tune.choice(config["space"]["batch"]),
            "max_steps": tune.choice(config["space"]["max_steps"]),
            "lr": tune.loguniform(config["space"]["lr"][0], config["space"]["lr"][1]),
            "scheduler_type": tune.choice(config["space"]["scheduler"]),
            "scheduler_warmup_proportion": tune.choice(config["space"]["scheduler_warmup_proportion"]),
        },
        resources_per_trial={"cpu": multiprocessing.cpu_count(), "gpu": torch.cuda.device_count()},
        num_samples=config["space"]["samples"]
    )
    print("Best config: ", analysis.get_best_config(metric="loss", mode="min"))
    print("RUN END")


def setup_logging(log_folder: str, experiment_name: str, enable: bool = True):
    """
    Setup of logger.

    :param log_folder: Folder where log file should be saved.
    :param experiment_name: Name of current experiment.
    :param enable: True enables logging. False disable.
    """
    if enable:
        timestamp = datetime.datetime.now().strftime("%Y-%m-%d_%H:%M:%S")
        stamp = timestamp + "_" + socket.gethostname()

        folder = os.path.join(log_folder, experiment_name)
        Path(folder).mkdir(parents=True, exist_ok=True)

        logging.basicConfig(format='%(process)d: %(levelname)s : %(asctime)s : %(message)s',
                            level=logging.DEBUG,
                            filename=os.path.join(folder, stamp + ".log"))
    else:
        logging.disable(logging.CRITICAL)


def kill_children():
    """
    Kills all subprocesses created by multiprocessing module.
    """

    for p in active_children():
        p.terminate()


def main():
    atexit.register(kill_children)
    args = ArgumentsManager.parse_args()

    if args is not None:
        args.func(args)
    else:
        exit(1)


if __name__ == '__main__':
    main()
