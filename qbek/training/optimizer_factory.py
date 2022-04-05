# -*- coding: UTF-8 -*-
""""
Created on 30.03.20
This module contains factory for creating optimizers.

:author:     Martin Dočekal
"""
from abc import ABC, abstractmethod
from typing import Union, Iterable, Callable, Dict

import torch


class OptimizerFactory(ABC):
    """
    Abstract base class for optimizers creation. (it's factory)
    """

    @abstractmethod
    def create(self, module: torch.nn.Module) -> torch.optim.Optimizer:
        """
        Creates optimizer for given module.

        :param module: Module which parameters you want to optimize.
        :type module: torch.nn.Module
        :return: Created optimizer for given module and with settings that are hold by factory.
        :rtype: torch.optim.Optimizer
        """
        pass

    @abstractmethod
    def create_for_params(self, params: Union[Iterable[torch.Tensor], Iterable[Dict]]) -> torch.optim.Optimizer:
        """
        Creates optimizer for given parameters.

        :param params: Parameters that should be optimized.
            An iterable of torch.Tensors or dicts which specifies which Tensors should be optimized along with group
            specific optimization options.
                Example of groups:
                    [
                        {'params': ..., 'weight_decay': ...},
                        {'params': ..., 'weight_decay': ...}
                    ]
        :type params: Union[Iterable[torch.Tensor], Iterable[Dict]]
        :return: Created optimizer for given params and with settings that are hold by factory.
        :rtype: torch.optim.Optimizer
        """
        pass


class AnyOptimizerFactory(OptimizerFactory):
    """
    Class that allows creation of any optimizer on demand.
    """

    def __init__(self, creator: Callable[..., torch.optim.Optimizer], attr: Dict, params_attr: str = "params"):
        """
        Initialization of factory.

        :param creator: This will be called with given attributes (attr) and the model parameters will be passed
            as paramsAttr attribute.
            You can use the class of optimizer itself.
        :type creator: Callable[..., torch.optim.Optimizer]
        :param attr: Dictionary with attributes that should be used. Beware that the attribute with name paramsAttr
            is reserved for model parameters.
        :type attr: Dict
        :param params_attr: Name of attribute that will be used to pass model parameters to optimizer.
        :type params_attr: str
        """

        self.creator = creator
        self.attr = attr
        self.params_attr = params_attr

    def create(self, module: torch.nn.Module) -> torch.optim.Optimizer:
        return self.create_for_params(module.parameters())

    def create_for_params(self, params: Union[Iterable[torch.Tensor], Iterable[Dict]]) -> torch.optim.Optimizer:
        self.attr[self.params_attr] = params
        return self.creator(**self.attr)
