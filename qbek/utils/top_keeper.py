# -*- coding: UTF-8 -*-
""""
Created on 02.11.21

Contains object for keeping top-k elements.

:author:     Martin Dočekal
"""
from typing import Tuple, Dict


class TopKeeper:
    """
    Keeps top k distinct elements.

    Example of usage:
        >>> k = TopKeeper(3)
        >>> k.push(1, "a")
        >>> k.push(2, "b")
        >>> k.push(3, "c")
        >>> k.push(0, "d")
        >>> list(k)
        ["c", "b", "a"]

        >>> k = TopKeeper(3)
        >>> k.push(1, "a")
        >>> k.push(2, "b")
        >>> k.push(3, "c")
        >>> k.push(10, "a")
        >>> list(k)
        ["a", "c", "b"]

    """

    def __init__(self, top: int):
        """
        initialization

        :param top: How many top elements will be maximally kept
        """
        assert top > 0
        self._top = top
        self._elements = {}
        self._sorted_elements = None
        self._min_score = None

    def __len__(self):
        return len(self._elements)

    def __getitem__(self, item: int):
        return self.get_element(item)[0]

    def score(self, item: int):
        """
        Score of element on given index.

        :param item: the index of an element
        """

        return self.get_element(item)[1]

    def get_element(self, item: int) -> Tuple[str, float]:
        """
        Get element from sorted sequence.

        :param item: Order of element (0 is the one with lowest score)
        :return: string value, score
        """

        try:
            return self._sorted_elements[item]
        except TypeError:
            self._sorted_elements = sorted(self._elements.items(), key=lambda x: x[1], reverse=True)
            return self._sorted_elements[item]

    def push(self, score: float, value: str):
        """
        Pushes new value into keeper.

        :param score: score of given value that determines
        :param value: value that should be stored
        """

        if len(self) < self._top or self._min_score is None or score > self._min_score:
            # this is not computationally cheap so we should do it only when we definitely want to insert
            self._sorted_elements = None    # invalidate it as scores will change
            self._min_score = self._push(self._elements, self._top, score, value)

    @staticmethod
    def _push(elements: Dict[str, float], top: int, score: float, value: str) -> float:
        if value in elements:
            # just replacement
            elements[value] = max(elements[value], score)
        else:
            elements[value] = score
            if len(elements) > top:
                del elements[min(elements, key=elements.get)]

        return min(elements.values())
