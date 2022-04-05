# -*- coding: UTF-8 -*-
""""
Created on 22.10.21

:author:     Martin Dočekal
"""

import unittest

from qbek.tokenizer import AlphaTokenizer


class TestAlphaTokenizer(unittest.TestCase):
    def setUp(self) -> None:
        self.tokenizer = AlphaTokenizer()

    def test_tokenize_empty(self):
        self.assertEqual(tuple(), self.tokenizer(""))

    def test_tokenize_non_striped(self):
        self.assertEqual(("Hello", "you", "."), self.tokenizer(" Hello you. "))

    def test_tokenize(self):
        self.assertEqual(("Hello", "how", "are", "you", "?"), self.tokenizer("Hello how are you?"))
        self.assertEqual(("300", "mm"), self.tokenizer("300 mm"))
        self.assertEqual(("Software", "simulation"), self.tokenizer("Software simulation"))
        self.assertEqual(
            ("conveyor", "-", "based", "continuous", "flow", "transport", "technology"),
            self.tokenizer("conveyor-based continuous flow transport technology"))
        self.assertEqual(
            ("car", "-", "based", "wafer", "-", "lot", "delivery"), self.tokenizer("car-based wafer-lot delivery")
        )
        self.assertEqual(("delivery", "time"), self.tokenizer("delivery time"))
        self.assertEqual(("throughput",), self.tokenizer("throughput"))


if __name__ == '__main__':
    unittest.main()
