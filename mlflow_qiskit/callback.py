"""Callback to log info from Qiskit to MLflow"""

from __future__ import annotations
import mlflow
from mlflow import log_metrics, log_params, log_text
from mlflow.utils.annotations import experimental
from qiskit_machine_learning.algorithms import SerializableModelMixin


@experimental
class MLflowCallback(SerializableModelMixin):
    """
    Callback class for logging Qiskit QML models.
    Currently, focusing on qiskit_machine_learning/neural_networks
    and quantum_image_processing/neural_networks.
    """

    def __init__(self, log_every_epoch=True):
        self.log_every_epoch = log_every_epoch

    def on_train_begin(self):
        setting = self.
