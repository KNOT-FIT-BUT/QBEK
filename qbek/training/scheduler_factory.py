# -*- coding: UTF-8 -*-
""""
Created on 30.03.20
This module contains factory for creating schedulers.

:author:     Martin Dočekal
"""

from abc import ABC, abstractmethod
from typing import Callable, Dict

import torch
from torch.optim.lr_scheduler import _LRScheduler   # TODO: protected member access, seems dirty :(


class SchedulerFactory(ABC):
    """
    Abstract base class for learning rate schedulers creation. (it's factory)
    """

    @abstractmethod
    def create(self, optimizer: torch.optim.Optimizer) -> _LRScheduler:
        """
        Creates scheduler for given optimizer.

        :param optimizer: The used optimizer that learning rate you want to schedule.
        :type optimizer: torch.optim.Optimizer
        :return: Created scheduler for given optimizer and with settings that are hold by factory.
        :rtype: torch.optim.Optimizer
        """
        pass


class AnySchedulerFactory(SchedulerFactory):
    """
    Class that allows creation of any scheduler on demand.
    """

    def __init__(self, creator: Callable[..., _LRScheduler], attr: Dict, optimizer_attr: str = "optimizer"):
        """
        Initialization of factory.

        :param creator: This will be called with given attributes (attr) and the optimizer will be passed
            as optimizerAttr attribute.
            You can use the class of scheduler itself.
        :type creator:  Callable[..., _LRScheduler]
        :param attr: Dictionary with attributes that should be used. Beware that the attribute with name optimizerAttr
            is reserved for optimizer.
        :type attr: Dict
        :param optimizer_attr: Name of attribute that will be used to pass optimizer to scheduler.
        :type optimizer_attr: str
        """

        self.creator = creator
        self.attr = attr
        self.optimizer_attr = optimizer_attr

    def create(self, optimizer: torch.optim.Optimizer) -> _LRScheduler:
        self.attr[self.optimizer_attr] = optimizer
        return self.creator(**self.attr)
