# QBEK
This is the official repository for the paper Query-Based Keyphrase Extraction from Long Documents containing the implementation of QBEK (Query-Based Extractor of Keyphrases).

# Installation
To install the extractor dependencies run:

    pip install -r requirements.txt

# Usage
To use the extractor, you first need a trained model (see [Training](#training) section). Next, you must set filters for it (see [Filters](#filters) section).
After selecting filters you can use it for extraction and evaluation (see [Evaluation](#evaluation) and [Extraction](#extraction) section).

## Training
There is a prepared toy example for training in the toy_examples folder.

Check the config whether it suits you and run the:

    run_train.sh

## Filters
There is a prepared toy example for evaluation in the toy_examples folder. But you need to firstly train a model.

You must first set the path to your model in the ["model"]["checkpoint"] field in the configuration. 

Then you can run the:

    run_filters.sh

## Evaluation
There is a prepared toy example for evaluation in the toy_examples folder. But you need to firstly train a model and set filters.

You must first set the path to your model in the ["model"]["checkpoint"] field in the configuration. 

Then you can run the:

    run_eval.sh

It first makes predictions and provides an alternative evaluation not published in our article. If you are interested in it, see the metrics package.

Then this script converts the predictions into a format suitable for the SemEval evaluator, and lastly, the SemEval evaluator is run.


## Extraction for evaluation
There is a prepared toy example for extraction in the toy_examples folder. But you need to firstly train a model and set filters.

It also provides inputs for the SemEval evaluation script as an evaluation tool but does not print the evaluation metrics defined in the metrics package.

You must first set the path to your model in the ["model"]["checkpoint"] field in the configuration. 

Then you can run the:

    run_extract.sh

# Configuration
All default configurations are saved in the config folder. You can find a description of configuration fields in the config files.

# Our trained models
If you are interested, you can download our already trained models. They have the filters already configured. All models use a title as a query.

| link | trained on | context size [sentences] |
|------|:------------:|:-------------:|
| https://knot.fit.vutbr.cz/QBEK/SemEval2010_query_19_sentences.ckpt | SemEval 2010 | 19|
| https://knot.fit.vutbr.cz/QBEK/Unstructured_SemEval2010_query_19_sentences.ckpt | Unstructured SemEval 2010 | 19 |
| https://knot.fit.vutbr.cz/QBEK/Library_query_19_sentences.ckpt | Library | 19 |
| https://knot.fit.vutbr.cz/QBEK/Inspec_query_1_sentence.ckpt | Inspec | 1 |

We trained a lot of variants, and there is only a selection of those. Please see our paper for more detail.

# Dataset format
The Dataset file that is used for training and evaluation is a jsonl file with the following fields:

* **uuid**: unique identifier of a document
* **query**: Query that is prepended to the context. We've used document title.
* **keyphrases**: List of keyphrases annotation for a given document.
* **contexts**: List of contexts that represents a document with keyphrase span annotations.
  * Every context contains the text itself and list of spans that are defined by start/end character offsets. 
  * Context and items in list of spans are separated by \t.
  * Example: 
    * This context contains keyphrase.\t5 12\t22 31
