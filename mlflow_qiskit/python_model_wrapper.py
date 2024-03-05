"""PythonModel wrapper for any Qiskit TrainableModel"""

from __future__ import annotations
from abc import ABCMeta
from typing import Optional, Any
from mlflow.pyfunc.model import PythonModel
from qiskit_machine_learning.algorithms import TrainableModel


class QuantumModel(PythonModel, metaclass=ABCMeta):
    """
    Subclass of PythonModel class that wraps various
    quantum models (currently, from Qiskit) as python
    models.
    """

    def __init__(self, model: TrainableModel):
        self.model = model

    def predict(
        self, context=None, model_input=None, params: Optional[dict[str, Any]] = None
    ):
        return self.model.predict(X=model_input)
