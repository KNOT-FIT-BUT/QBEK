# -*- coding: UTF-8 -*-
""""
Created on 09.04.21

:author:     Martin Dočekal
"""
import os
import unittest
from pathlib import Path
from shutil import copyfile

import torch
from transformers import AutoTokenizer

from qbek.batch import Batch
from qbek.datasets.czech_library import CzechLibrarySampleSplitter, CzechLibrary, CzechLibraryDataModule


class TestCzechLibrarySampleSplitter(unittest.TestCase):
    def setUp(self) -> None:
        self.tokenizer = AutoTokenizer.from_pretrained("bert-base-multilingual-cased")
        self.tokenizer.model_max_length = 7
        self.splitter = CzechLibrarySampleSplitter(self.tokenizer)

    def test_join_spans(self):
        self.assertTrue(self.splitter.join_spans([]) == set())
        self.assertTrue(self.splitter.join_spans([(1, 10), (11, 11)]) == {(1, 10), (11, 11)})
        self.assertTrue(self.splitter.join_spans([(1, 10), (10, 11), (15, 20), (7, 12)]) == {(1, 12), (15, 20)})
        self.assertTrue(self.splitter.join_spans([(1, 10), (5, 7)]) == {(1, 10)})
        self.assertTrue(self.splitter.join_spans([(1, 10), (10, 100), (100, 101)]) == {(1, 101)})
        self.assertTrue(
            self.splitter.join_spans([(1, 10), (10, 100), (100, 101), (200, 201)]) == {(1, 101), (200, 201)})

    def test_fair_split(self):
        self.assertListEqual([(0, 10), (11, 30)],
                             self.splitter.fair_split(2, [(0, 10), (11, 20), (23, 30)], [(11, 30)]))

        self.assertListEqual([(0, 10), (11, 30)],
                             self.splitter.fair_split(2, [(0, 10), (11, 20), (23, 30)], [(0, 10), (11, 30)]))

    def test_split(self):
        """
        input string - un-tokenized input
        tokenized - tokenized input string that should be put on the input of a model
        tokens offset mapping
        keyphrase spans as right exclusive token offsets
        """
        res = list(self.splitter.split("t", "fits"))[0]

        self.assertListEqual([101, 188, 102, 21635, 10107, 102], res[0].tolist() + res[1].tolist())
        self.assertListEqual([], res[2].tolist())

        tokens = [
            [101, 12887, 102, 27775, 11695, 28615, 102],
            [101, 12887, 102, 10114, 10347, 24137, 102],
            [101, 12887, 102, 119, 102]
        ]

        spans = [[], [], []]
        res = list(self.splitter.split("title", "Too long needs to be split."))
        self.assertEqual(3, len(res))

        for i, r in enumerate(res):
            self.assertListEqual(tokens[i], r[0].tolist() + r[1].tolist())
            self.assertListEqual(spans[i], r[2].tolist())

        res = list(self.splitter.split("", "Too long needs to be"))[0]

        self.assertIsNone(res[0])
        self.assertListEqual([101, 27775, 11695, 28615, 10114, 10347, 102], res[1].tolist())
        self.assertListEqual([], res[2].tolist())

        tokens = [
            [101, 12887, 102, 27775, 11695, 102],
            [101, 12887, 102, 28615, 10114, 10347, 102],
            [101, 12887, 102, 24137, 119, 102]
        ]

        spans = [[], [[3, 4]], []]
        res = list(self.splitter.split("title", "Too long needs to be split.\t9 17"))
        self.assertEqual(3, len(res))

        for i, r in enumerate(res):
            self.assertListEqual(tokens[i], r[0].tolist() + r[1].tolist())
            self.assertListEqual(spans[i], r[2].tolist())

        with self.assertRaises(RuntimeError):
            self.splitter.split("title", "Too long needs to be split.\t0 27")

    def test_split_real_problem_i_have_had(self):
        self.splitter._tokenizer.model_max_length = 512
        title = "Katalog lékařů a zdravotnických zařízení ... : Česká republika"
        title_tokens = self.tokenizer.encode(title)
        context = "é H-32 ské K-32 ách A-72 jí K-63 líce A-58 líce B-30 líce D-39 líce E-16 líce F-29 lice G-12 lice G-28 líce H-39 líce 1-16 nice 1-24 líce 1-33 lice J-24 líce J-4 nice K-48 líce K-54 líce K-74 lice L-19 líce 0-31 líce 0-43 líce 0-6 K-63 ách M-10 ké D-8 u H-10 iech I-33 ci G-6 íníku A-95 B-48 C-31 iku 1-16 B-42 ské D-20 J-17 H-39 J-24 0-70 L-39 C-15 I I-24 i G-12 B-5 F-29 1 K-32 -16 0-6 F-35 ä D-32 B-5 Ihu A-43 H-10 Lékárna Lékárna Lékárna Lékárna Lékárna Lékárna Lékárna Lékárna Lékárna Lékárna Lékárna Lékárna Lékárna Lékárna Lékárna Lékárna Lékárna Lékárna Lékárna Lékárna Lékárna Lékárna Lékárna Lékárna Lékárna Lékárna Lékárna Lékárna Lékárna Lékárna Lékárna Lékárna Lékárna Lékárna Lékárna Lékárna Lékárna Lékárna Lékárna Lékárna Lékárna Lékárna Lékárna Lékárna Lékárna Lékárna Lékárna Lékárna Lékárna Lékárna Lékárna Lékárna Lékárna Lékárna Lékárna Lékárna Lékárna Lékárna Lékárna Lékárna Lékárna Lékárna Lékárna Lékárna Lékárna Lékárna Lékárna Lékárna Lékárna Lékárna Lékárna Lékárna Lékárna Lékárna Lékárna Lékárna Lékárna Lékárna Lékárna Lékárna Lékárna Lékárna Lékárna Lékárna Lékárna Lékárna Lékárna Lékárna Lékárna Na Žižkově A-28 Nad Knížecí A-58 Nad Muzeem A-19 Nad Primaskou A-106 Nad řekou C-26 Naděje A-43."
        context_tokens = self.tokenizer.encode(context)[1:-1]
        context_and_spans = context + "	418 425	426 433	434 441	442 449	450 457	458 465	466 473	474 481	482 489	490 497	498 505	506 513	514 521	522 529	530 537	538 545	546 553	554 561	562 569	570 577	578 585	586 593	594 601	602 609	610 617	618 625	626 633	634 641	642 649	650 657	658 665	666 673	674 681	682 689	690 697	698 705	706 713	714 721	722 729	730 737	738 745	746 753	754 761	762 769	770 777	778 785	786 793	794 801	802 809	810 817	818 825	826 833	834 841	842 849	850 857	858 865	866 873	874 881	882 889	890 897	898 905	906 913	914 921	922 929	930 937	938 945	946 953	954 961	962 969	970 977	978 985	986 993	994 1001	1002 1009	1010 1017	1018 1025	1026 1033	1034 1041	1042 1049	1050 1057	1058 1065	1066 1073	1074 1081	1082 1089	1090 1097	1098 1105	1106 1113	1114 1121	1122 1129"

        parts = self.splitter.split(title, context_and_spans)

        assembled_context = []
        for p in parts:
            act_tokens = p[0].tolist() + p[1].tolist()
            sep_offset = act_tokens.index(self.splitter._tokenizer.sep_token_id)
            act_title = act_tokens[:sep_offset + 1]
            act_context = act_tokens[sep_offset + 1:-1]
            self.assertEqual(title_tokens, act_title)
            assembled_context += act_context
        self.assertListEqual(context_tokens, assembled_context)

    def test_split_too_long_title(self):
        res = list(self.splitter.split("Too long title so this will not be split at all.",
                                       "Ordinary context string."))[0]
        self.assertListEqual(
            [101, 27775, 11695, 12887, 10380, 10531, 11337, 10472, 10347, 24137, 10160, 10435, 119, 102, 19372, 25755,
             10908, 30798, 33714, 119, 102], res[0].tolist() + res[1].tolist())


class TestCzechLibrary(unittest.TestCase):
    path_to_this_script_file = os.path.dirname(os.path.realpath(__file__))
    dataset_path = os.path.join(path_to_this_script_file, "fixtures/dataset.txt")
    tmp_path = os.path.join(path_to_this_script_file, "tmp/")
    dataset_path_tmp = os.path.join(path_to_this_script_file, "tmp/dataset.txt")
    dataset_prep_path_title_tmp = \
        os.path.join(path_to_this_script_file, "tmp/dataset_bert-base-multilingual-cased_title.prep")
    dataset_prep_path_no_title_tmp = \
        os.path.join(path_to_this_script_file, "tmp/dataset_bert-base-multilingual-cased_no_title.prep")

    dataset_different_prep_path = os.path.join(path_to_this_script_file,
                                               "fixtures/dataset_bert-base-multilingual-cased_title.prep")
    dataset_different_prep_path_tmp = os.path.join(path_to_this_script_file,
                                                   "tmp/dataset_bert-base-multilingual-cased_title.prep")

    dataset_lines = [
        """{"uuid": "uuid:64d1e9a0-e7c9-11e6-9964-005056825209", "query": "opera 1", "keyphrases": ["opera", "příběh"], "contexts": ["BÍLA pani Komická opera o 3 dějstvích.\\t18 23", "Nauč se vyprávět příběh o narození Ježíška.\\t17 23"]}""",
        """{"uuid": "uuid:64d1e9a0-e7c9-11e6-9964-005056825211", "query": "Title 2", "keyphrases": ["hudby", "divadla"], "contexts": ["To ovšem u spontánní rockové hudby příliš v úvahu nepřipadá.\\t29 34", "srpna 1881 ve prospěch obnovené stavby Národního divadla p.\\t49 56", "Smotala jsem, co mi kdo vyprávěl, s vlastní vzpomínkou."]}""",
        """{"uuid": "uuid:64d1e9a0-e7c9-11e6-9964-005056825212", "query": "Title 3", "keyphrases": ["Klášterní", "kostel"], "contexts": ["Klášterní kostel Nanebevzetí P. Marie-tympanon západního portálu.\\t0 9\\t10 16"]}"""
    ]

    dataset_lines_offsets = [0, 243, 585]

    def setUp(self) -> None:
        copyfile(self.dataset_path, self.dataset_path_tmp)
        self.tokenizer = AutoTokenizer.from_pretrained("bert-base-multilingual-cased")

    def tearDown(self) -> None:
        # remove all in tmp but placeholder
        for f in Path(self.tmp_path).glob('*'):
            if not str(f).endswith("placeholder"):
                os.remove(f)

    def test_init_with_index(self):
        dataset = CzechLibrary(self.dataset_path_tmp, self.tokenizer, verbose=False)
        self.assertTrue(os.path.isfile(self.dataset_prep_path_title_tmp))

        self.assertEqual(len(dataset.preprocessed), 6)
        self.assertEqual(dataset.preprocessed._index["original_dataset_size"], os.path.getsize(self.dataset_path_tmp))

    def test_init_with_different_index(self):
        copyfile(self.dataset_different_prep_path, self.dataset_different_prep_path_tmp)
        copyfile(self.dataset_different_prep_path + ".index", self.dataset_different_prep_path_tmp + ".index")

        with self.assertRaises(RuntimeError):
            _ = CzechLibrary(self.dataset_path_tmp, self.tokenizer, verbose=False)

    def test_len(self):
        dataset = CzechLibrary(self.dataset_path_tmp, self.tokenizer, verbose=False)
        self.assertEqual(len(dataset), 6)

    def test_preprocess(self):
        prep_dataset = CzechLibrary.preprocess(self.dataset_path_tmp, self.tokenizer, False, verbose=False)
        self.assertTrue(os.path.isfile(self.dataset_prep_path_no_title_tmp))
        self.assertTrue(os.path.isfile(self.dataset_prep_path_no_title_tmp + ".index"))
        self.assertEqual(len(prep_dataset), 6)
        self.assertEqual(prep_dataset._index["original_dataset_size"], os.path.getsize(self.dataset_path_tmp))

    def test_path_to_prep_for_dataset(self):
        self.assertEqual(CzechLibrary.path_to_preprocessed_dataset(self.dataset_path_tmp, self.tokenizer, title=True),
                         self.dataset_prep_path_title_tmp)

        self.assertEqual(CzechLibrary.path_to_preprocessed_dataset(self.dataset_path_tmp, self.tokenizer, title=False),
                         self.dataset_prep_path_no_title_tmp)

    def test_path_to_preprocessed(self):
        dataset = CzechLibrary(self.dataset_path_tmp, self.tokenizer, verbose=False)
        self.assertEqual(dataset.path_to_preprocessed, self.dataset_prep_path_title_tmp)

    def test_with(self):
        dataset = CzechLibrary(self.dataset_path_tmp, self.tokenizer, verbose=False)
        self.assertIsNone(dataset._file_descriptor)
        with dataset:
            self.assertIsNotNone(dataset._file_descriptor)
        self.assertIsNone(dataset._file_descriptor)

    def test_open_close(self):
        dataset = CzechLibrary(self.dataset_path_tmp, self.tokenizer, verbose=False)
        self.assertIsNone(dataset._file_descriptor)
        dataset.open()
        self.assertIsNotNone(dataset._file_descriptor)
        dataset.close()
        self.assertIsNone(dataset._file_descriptor)

    def test_line(self):
        with CzechLibrary(self.dataset_path_tmp, self.tokenizer, verbose=False) as dataset:
            for i, (offset, line) in enumerate(zip(self.dataset_lines_offsets, self.dataset_lines)):
                self.assertEqual(dataset.line(offset), line)

    def test_part(self):
        with CzechLibrary(self.dataset_path_tmp, self.tokenizer, verbose=False) as dataset:
            self.assertEqual(dataset.part(126, 49), "BÍLA pani Komická opera o 3 dějstvích.	18 23")

    def test_transform_spans(self):
        self.assertEqual(CzechLibrary.transform_spans([]), [])
        self.assertEqual(CzechLibrary.transform_spans(["0 1", "2 3"]), [(0, 1), (2, 3)])

    def test_get_item(self):
        with CzechLibrary(self.dataset_path_tmp, self.tokenizer, verbose=False) as dataset:
            line_offset, tokens, spans = dataset[0]
            self.assertEqual(0, line_offset)
            self.assertListEqual(
                [101, 13335, 122, 102, 139, 110914, 44376, 97586, 106306, 45007, 13335, 183, 124, 172, 11636, 73717,
                 24204, 10269, 119, 102],
                tokens.tolist())
            self.assertEqual([[10, 10]], spans.tolist())

    def test_char_2_token_lvl_span(self):
        offset_mapping = [(0, 0), (0, 1), (1, 2), (2, 4), (5, 9), (10, 13), (13, 17), (18, 23), (24, 25), (26, 27),
                          (28, 29), (29, 30), (30, 33), (33, 35), (35, 37), (37, 38), (0, 0)]

        self.assertEqual(CzechLibrary.char_2_token_lvl_span((2, 5), offset_mapping), (3, 4))
        self.assertEqual(CzechLibrary.char_2_token_lvl_span((0, 1), offset_mapping), (1, 1))
        self.assertEqual(CzechLibrary.char_2_token_lvl_span((18, 30), offset_mapping), (7, 11))
        self.assertEqual(CzechLibrary.char_2_token_lvl_span((0, 38), offset_mapping), (1, 15))
        self.assertEqual(CzechLibrary.char_2_token_lvl_span((3, 8), offset_mapping), (3, 4))

        with self.assertRaises(RuntimeError):
            CzechLibrary.char_2_token_lvl_span((999, 9999), offset_mapping)

    def test_collate_fn(self):
        dataset = CzechLibrary(self.dataset_path_tmp, self.tokenizer, verbose=False)
        # (line_offset, tokens, spans)
        samples = [
            (0, torch.tensor([1, 1, 1, 1, 1]), torch.tensor([])),
            (256, torch.tensor([2, 2, 2, 2]), torch.tensor([[1, 2]])),
            (512, torch.tensor([3, 3, 3, 3, 3]), torch.tensor([[3, 3]]))
        ]
        batch_tokens = torch.tensor([
            [1, 1, 1, 1, 1],
            [2, 2, 2, 2, self.tokenizer.pad_token_id],
            [3, 3, 3, 3, 3]
        ])

        batch_attention_mask = torch.ones_like(batch_tokens)
        batch_attention_mask[1, 4] = 0

        batch_gt_span_universe = torch.zeros((batch_tokens.shape[0], batch_tokens.shape[1], batch_tokens.shape[1]),
                                             dtype=torch.bool)
        batch_gt_span_universe[1][1][2] = True
        batch_gt_span_universe[2][3][3] = True

        batch_line_offsets = [0, 256, 512]

        batch = dataset.collate_fn(samples)
        self.assertListEqual(batch.tokens.tolist(), batch_tokens.tolist())
        self.assertListEqual(batch.attention_masks.tolist(), batch_attention_mask.tolist())
        self.assertListEqual(batch.gt_span_universes.tolist(), batch_gt_span_universe.tolist())
        self.assertListEqual(batch.line_offsets, batch_line_offsets)

    def test_collate_fn_truncate(self):
        dataset = CzechLibrary(self.dataset_path_tmp, self.tokenizer, verbose=False)
        # (line_offset, tokens, spans)
        input_tokens = torch.randint(0, 30000, (513,))
        samples = [
            (0, input_tokens, torch.tensor([])),
        ]
        batch_tokens = torch.tensor([
            input_tokens[:512].tolist()
        ])

        batch_attention_mask = torch.ones_like(batch_tokens)

        batch_gt_span_universe = torch.zeros((batch_tokens.shape[0], batch_tokens.shape[1], batch_tokens.shape[1]),
                                             dtype=torch.bool)

        batch_line_offsets = [0]

        batch = dataset.collate_fn(samples)
        self.assertListEqual(batch.tokens.tolist(), batch_tokens.tolist())
        self.assertListEqual(batch.attention_masks.tolist(), batch_attention_mask.tolist())
        self.assertListEqual(batch.gt_span_universes.tolist(), batch_gt_span_universe.tolist())
        self.assertListEqual(batch.line_offsets, batch_line_offsets)

    def test_collate_fn_without_doc_emb(self):
        dataset = CzechLibrary(self.dataset_path_tmp, self.tokenizer, verbose=False)
        samples = [
            (0, torch.tensor([1, 1, 1, 1, 1]), torch.tensor([])),
            (256, torch.tensor([2, 2, 2, 2]), torch.tensor([[1, 2]])),
            (512, torch.tensor([3, 3, 3, 3, 3]), torch.tensor([[3, 3]]))
        ]
        batch_tokens = torch.tensor([
            [1, 1, 1, 1, 1],
            [2, 2, 2, 2, self.tokenizer.pad_token_id],
            [3, 3, 3, 3, 3]
        ])

        batch_attention_mask = torch.ones_like(batch_tokens)
        batch_attention_mask[1, 4] = 0

        batch_gt_span_universe = torch.zeros((batch_tokens.shape[0], batch_tokens.shape[1], batch_tokens.shape[1]),
                                             dtype=torch.bool)
        batch_gt_span_universe[1][1][2] = True
        batch_gt_span_universe[2][3][3] = True

        batch_line_offsets = [0, 256, 512]

        batch = dataset.collate_fn(samples)
        self.assertListEqual(batch.tokens.tolist(), batch_tokens.tolist())
        self.assertListEqual(batch.attention_masks.tolist(), batch_attention_mask.tolist())
        self.assertListEqual(batch.gt_span_universes.tolist(), batch_gt_span_universe.tolist())
        self.assertListEqual(batch.line_offsets, batch_line_offsets)


class TestCzechLibraryDataModule(unittest.TestCase):
    path_to_this_script_file = os.path.dirname(os.path.realpath(__file__))
    dataset_prep_path = os.path.join(path_to_this_script_file,
                                     "fixtures/dataset_bert-base-multilingual-cased_title.prep")
    dataset_index_path = os.path.join(path_to_this_script_file,
                                      "fixtures/dataset_bert-base-multilingual-cased_title.prep.index")

    train_path = os.path.join(path_to_this_script_file, "fixtures/train.txt")
    train_path_tmp = os.path.join(path_to_this_script_file, "tmp/train.txt")
    train_prep_path = os.path.join(path_to_this_script_file, "fixtures/train.prep")
    train_index_path = os.path.join(path_to_this_script_file, "fixtures/train.prep.index")
    train_prep_path_tmp = os.path.join(path_to_this_script_file,
                                       "tmp/train_bert-base-multilingual-cased_title.prep")
    train_index_path_tmp = os.path.join(path_to_this_script_file,
                                        "tmp/train_bert-base-multilingual-cased_title.prep.index")

    val_path = os.path.join(path_to_this_script_file, "fixtures/val.txt")
    val_path_tmp = os.path.join(path_to_this_script_file, "tmp/val.txt")
    val_prep_path_tmp = os.path.join(path_to_this_script_file,
                                     "tmp/val_bert-base-multilingual-cased_title.prep")
    val_index_path_tmp = os.path.join(path_to_this_script_file,
                                      "tmp/val_bert-base-multilingual-cased_title.prep.index")

    test_path = os.path.join(path_to_this_script_file, "fixtures/test.txt")
    test_path_tmp = os.path.join(path_to_this_script_file, "tmp/test.txt")
    test_prep_path_tmp = os.path.join(path_to_this_script_file,
                                      "tmp/test_bert-base-multilingual-cased_title.prep")
    test_index_path_tmp = os.path.join(path_to_this_script_file,
                                       "tmp/test_bert-base-multilingual-cased_title.prep.index")

    tmp_path = os.path.join(path_to_this_script_file, "tmp/")

    def setUp(self) -> None:
        self.config = {
            "training": {
                "activate": True,
                "dataset": {
                    "path": self.train_path_tmp,
                    "workers": 0,
                    "prep_workers": 0,
                    "add_title": True
                },
                "batch": 8
            },
            "doc_emb": {
                "type": None,
                "resume": None,
                "max_features": None
            },
            "validation": {
                "activate": True,
                "dataset": {
                    "path": self.val_path_tmp,
                    "workers": 0,
                    "prep_workers": 0,
                    "add_title": True
                },
                "batch": 8
            },
            "testing": {
                "activate": True,
                "dataset": {
                    "path": self.test_path_tmp,
                    "workers": 0,
                    "prep_workers": 0,
                    "add_title": True
                },
                "batch": 8
            },
            "transformers": {
                "tokenizer": "bert-base-multilingual-cased",
                "cache": None,
                "local_files_only": False
            }

        }

        copyfile(self.train_path, self.train_path_tmp)
        copyfile(self.val_path, self.val_path_tmp)
        copyfile(self.test_path, self.test_path_tmp)

        self.data_module = CzechLibraryDataModule(self.config)

    def tearDown(self) -> None:
        # remove all in tmp but placeholder
        for f in Path(self.tmp_path).glob('*'):
            if not str(f).endswith("placeholder"):
                os.remove(f)

        self.data_module.teardown()

    def test_init(self):
        self.assertEqual(self.data_module.config, self.config)
        self.assertIsNone(self.data_module.train)
        self.assertIsNone(self.data_module.val)
        self.assertIsNone(self.data_module.test)
        self.assertEqual(self.data_module.tokenizer.name_or_path, self.config["transformers"]["tokenizer"])

    def test_train_dataloader_no_setup(self):
        with self.assertRaises(Exception):
            self.data_module.train_dataloader()

    def test_train_dataloader(self):
        self.data_module.setup()
        loader = self.data_module.train_dataloader()
        self.assertEqual(loader.batch_size, 8)
        self.assertTrue(loader.pin_memory)
        self.assertEqual(loader.dataset.path_to, self.train_path_tmp)

    def test_val_dataloader_no_setup(self):
        with self.assertRaises(Exception):
            self.data_module.val_dataloader()

    def test_val_dataloader(self):
        self.data_module.setup()
        loader = self.data_module.val_dataloader()
        self.assertEqual(loader.batch_size, 8)
        self.assertTrue(loader.pin_memory)
        self.assertEqual(loader.dataset.path_to, self.val_path_tmp)

    def test_test_dataloader_no_setup(self):
        with self.assertRaises(Exception):
            self.data_module.test_dataloader()

    def test_test_dataloader(self):
        self.data_module.setup()
        loader = self.data_module.test_dataloader()
        self.assertEqual(loader.batch_size, 8)
        self.assertTrue(loader.pin_memory)
        self.assertEqual(loader.dataset.path_to, self.test_path_tmp)

    def test_prepare_data(self):
        self.assertFalse(os.path.isfile(self.train_prep_path_tmp))
        self.assertFalse(os.path.isfile(self.val_prep_path_tmp))
        self.assertFalse(os.path.isfile(self.test_prep_path_tmp))
        self.assertFalse(os.path.isfile(self.train_index_path_tmp))
        self.assertFalse(os.path.isfile(self.val_index_path_tmp))
        self.assertFalse(os.path.isfile(self.test_index_path_tmp))

        self.data_module.prepare_data()

        self.assertTrue(os.path.isfile(self.train_prep_path_tmp))
        self.assertTrue(os.path.isfile(self.val_prep_path_tmp))
        self.assertTrue(os.path.isfile(self.test_prep_path_tmp))
        self.assertTrue(os.path.isfile(self.train_index_path_tmp))
        self.assertTrue(os.path.isfile(self.val_index_path_tmp))
        self.assertTrue(os.path.isfile(self.test_index_path_tmp))

    def test_prepare_data_train_load(self):
        copyfile(self.train_prep_path, self.train_prep_path_tmp)
        copyfile(self.train_index_path, self.train_index_path_tmp)
        self.data_module.prepare_data()

        self.assertTrue(os.path.isfile(self.train_prep_path_tmp))
        self.assertTrue(os.path.isfile(self.val_prep_path_tmp))
        self.assertTrue(os.path.isfile(self.test_prep_path_tmp))
        self.assertTrue(os.path.isfile(self.train_index_path_tmp))
        self.assertTrue(os.path.isfile(self.val_index_path_tmp))
        self.assertTrue(os.path.isfile(self.test_index_path_tmp))

    def test_prepare_data_invalid_train_index_load(self):
        copyfile(self.dataset_prep_path, self.train_prep_path_tmp)
        copyfile(self.dataset_index_path, self.train_index_path_tmp)
        with self.assertRaises(RuntimeError):
            self.data_module.prepare_data()

    def test_prepare_data_invalid_val_index_load(self):
        copyfile(self.dataset_prep_path, self.val_prep_path_tmp)
        copyfile(self.dataset_index_path, self.val_index_path_tmp)
        with self.assertRaises(Exception):
            self.data_module.prepare_data()

    def test_prepare_data_invalid_test_index_load(self):
        copyfile(self.dataset_prep_path, self.test_prep_path_tmp)
        copyfile(self.dataset_index_path, self.test_index_path_tmp)
        with self.assertRaises(Exception):
            self.data_module.prepare_data()

    def test_setup(self):
        self.data_module.setup()
        self.assertTrue(self.data_module.train.path_to, self.train_path_tmp)
        self.assertTrue(self.data_module.val.path_to, self.val_path_tmp)
        self.assertTrue(self.data_module.test.path_to, self.test_path_tmp)

    def test_teardown_on_closed(self):
        # should be empty op without errors
        self.data_module.teardown()

    def test_teardown(self):
        # should be without errors
        self.data_module.setup()
        self.data_module.teardown()
        self.assertIsNone(self.data_module.train._file_descriptor)
        self.assertIsNone(self.data_module.val._file_descriptor)
        self.assertIsNone(self.data_module.test._file_descriptor)

    def test_transfer_batch_to_device(self):
        if torch.cuda.is_available():
            device = torch.device("cuda:0")
            cpu_device = torch.device("cpu")
            tokens = torch.rand(10, 10)
            attention_masks = torch.rand(10, 10)
            gt_span_universes = torch.rand(2, 10, 10)
            line_offsets = [0]
            tokenizer = AutoTokenizer.from_pretrained("bert-base-multilingual-cased")
            batch = Batch(tokens, attention_masks, gt_span_universes, line_offsets, tokenizer)

            batch_on_device = batch.to(device)

            self.assertEqual(batch.tokens.device, cpu_device)
            self.assertEqual(batch.attention_masks.device, cpu_device)
            self.assertEqual(batch.gt_span_universes.device, cpu_device)

            self.assertEqual(batch_on_device.tokens.device, device)
            self.assertEqual(batch_on_device.attention_masks.device, device)
            self.assertEqual(batch_on_device.gt_span_universes.device, device)

            self.assertListEqual(batch.tokens.tolist(), batch_on_device.tokens.tolist())
            self.assertListEqual(batch.attention_masks.tolist(), batch_on_device.attention_masks.tolist())
            self.assertListEqual(batch.gt_span_universes.tolist(), batch_on_device.gt_span_universes.tolist())
            self.assertListEqual(batch.line_offsets, batch_on_device.line_offsets)

        else:
            self.skipTest("Cuda device is not available.")


if __name__ == '__main__':
    unittest.main()
