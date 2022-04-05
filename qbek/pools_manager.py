# -*- coding: UTF-8 -*-
""""
Created on 12.04.21

:author:     Martin Dočekal
"""
import multiprocessing

from typing import Dict, Any


class PoolsManager:
    """
    Manages all multiprocessing pools.
    """

    def __init__(self, config: Dict[str, Any]):
        """
        Initialization of manager.

        :param config: user's configuration
        """
        self._config = config
        self._auto_filter = None

    @property
    def auto_filter(self) -> multiprocessing.Pool:
        """
        Workers pool for auto filter.
        """
        return self._auto_filter

    @property
    def auto_filter_workers(self) -> int:
        """
        Number of workers in auto filter pool.
        """
        return self._config["filters"]["workers"]

    def __enter__(self):
        self.open()

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()

    def open(self):
        """
        Opens all pools.
        """
        self.close()
        if self._config["filters"]["workers"] > 0:
            self._auto_filter = multiprocessing.Pool(processes=self.auto_filter_workers)

    def close(self):
        """
        closes all pools
        """
        if self._auto_filter is not None:
            self._auto_filter.close()
            self._auto_filter.join()
            self._auto_filter = None
