# -*- coding: UTF-8 -*-
""""
Created on 02.11.21

:author:     Martin Dočekal
"""

import unittest

from qbek.utils.top_keeper import TopKeeper


class TestTopKeeperInit(unittest.TestCase):

    def test_init(self):
        TopKeeper(1)

    def test_init_fail(self):
        with self.assertRaises(AssertionError):
            TopKeeper(-10)
        with self.assertRaises(AssertionError):
            TopKeeper(0)


class TestTopKeeper(unittest.TestCase):

    def setUp(self) -> None:
        self.keeper = TopKeeper(3)

    def test_len_priority(self):
        self.keeper.push(10, "a")
        self.keeper.push(100, "b")
        self.keeper.push(1000, "c")
        self.keeper.push(10000, "d")
        self.assertEqual(len(self.keeper), 3)

    def test_len_distinct(self):
        self.keeper.push(10, "a")
        self.keeper.push(100, "a")
        self.keeper.push(1000, "a")
        self.keeper.push(10000, "a")
        self.assertEqual(len(self.keeper), 1)

    def test_push_priority_in_order(self):
        self.keeper.push(10, "a")
        self.keeper.push(100, "b")
        self.keeper.push(1000, "c")
        self.keeper.push(10000, "d")
        self.assertEqual(["d", "c","b"], list(self.keeper))
        self.assertEqual([10000, 1000, 100], list(self.keeper.score(i) for i in range(3)))

    def test_push_priority_not_in_order(self):
        self.keeper.push(100, "b")
        self.keeper.push(10, "a")
        self.keeper.push(10000, "d")
        self.keeper.push(1000, "c")
        self.assertEqual(["d", "c", "b"], list(self.keeper))
        self.assertEqual([10000, 1000, 100], list(self.keeper.score(i) for i in range(3)))

    def test_push_distinct(self):
        self.keeper.push(10, "a")
        self.keeper.push(100, "b")
        self.keeper.push(1000, "c")
        self.keeper.push(1500, "a")
        self.keeper.push(10000, "d")

        self.assertEqual(["d", "a", "c"], list(self.keeper))
        self.assertEqual([10000, 1500, 1000], list(self.keeper.score(i) for i in range(3)))

        self.keeper.push(90000, "a")

        self.assertEqual(["a", "d", "c"], list(self.keeper))
        self.assertEqual([90000, 10000, 1000], list(self.keeper.score(i) for i in range(3)))

    def test_push_distinct_no_change(self):
        self.keeper.push(10, "a")
        self.keeper.push(100, "b")
        self.keeper.push(1000, "c")
        self.keeper.push(5, "a")

        self.assertEqual(["c", "b", "a"], list(self.keeper))
        self.assertEqual([1000, 100, 10], list(self.keeper.score(i) for i in range(3)))


if __name__ == '__main__':
    unittest.main()
