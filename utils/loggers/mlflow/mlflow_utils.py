"""Main Logger class for MLflow experiment tracking."""
import os
import logging
import re
from pathlib import Path
from typing import Any, List, Dict
from argparse import Namespace

from utils.general import colorstr

try:
    import mlflow
    import mlflow.pytorch
    from mlflow.exceptions import RestException

    assert hasattr(mlflow, "__version__")
except (ImportError, AssertionError):
    mlflow = None

MLFLOW_TRACKING_URI = os.environ.get("MLFLOW_TRACKING_URI")
REGEX_METRICS = re.compile(r"[^a-zA-Z0-9\/\_\-\. ]")
PREFIX = colorstr("MLflow: ")

logger = logging.getLogger(__name__)


class MlflowTrackerNotSetError(Exception):
    """
    Indicates that the MLflow tracking URL is not set. When running
    as part of an Azure ML job, the tracking URL is set automatically.
    When using a stand-alone MLflow server, the MLFLOW_TRACKING_URI
    environment variable must be set.

    Attributes:
        msg (str): Human readable string describing the exception.
    """

    def __init__(self):
        self.msg = (
            f"{PREFIX}WARNING ⚠️ MLflow is installed but not configured, "
            "the environment variable MLFLOW_TRACKING_URI must be set. "
            "Skipping Mlflow logging."
        )
        super().__init__(self.msg)


class MlflowLogger:
    """
    Log training runs, parameters and models to MLflow or Azure ML.

    Note:
        When running an Azure ML Job, an MLflow run is automatically setup (including
        the tracking URL). The metrics, images and outputs will be available in the
        Job GUI without any additional setups.

        When using a stand-alone version of MLflow, the tracking URL of the server
        must be set by creating an MLFLOW_TRACKING_URI environment variable.

        Example: export MLFLOW_TRACKING_URI=https://my-mlflow-tracker.net

    Attributes:
        run_tracking_url (str): The URL to track the current run.
        is_azure_ml (bool): True if the logging run into an Azure ML Job.

    Raises:
        MlflowTrackerNotSetError: The Mlflow Tracking URL is not set and the run is
        not part of an Azure ML Job.
    """

    def __init__(self, opt: Namespace):
        if mlflow:
            self.run_tracking_url = ""
            self.mlflow = mlflow
            self.is_azure_ml = self.mlflow.get_tracking_uri().startswith("azureml://")

            if not self.is_azure_ml:
                if MLFLOW_TRACKING_URI:
                    self.mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)

                if not self.mlflow.get_tracking_uri().startswith(
                    ("http://", "https://")
                ):
                    raise MlflowTrackerNotSetError()

                self.mlflow.start_run()

            active_run = self.mlflow.active_run()
            if active_run:
                self.run_tracking_url = (
                    f"{self.mlflow.get_tracking_uri()}/#/experiments/"
                    f"{active_run.info.experiment_id}/runs/{active_run.info.run_id}"
                )

            self._set_initial_params(opt)

    def _set_initial_params(self, args: Namespace):
        """
        Log the initial set of parameters for the current MLflow run.

        Note:
            Only the args listed in *included_params* are logged.

        Args:
            args (argparse.Namespace): A Namespace containing the arguments passed
                to the process.
        """
        included_params = [
            "weights",
            "data",
            "hyp",
            "epochs",
            "imgsz",
            "batch_size",
            "evolve",
            "freeze",
            "save_period",
        ]
        initial_params = self._sanitize_dictionary(
            {k: v for k, v in vars(args).items() if k in included_params}
        )

        if "weights" in initial_params:
            initial_params["weights"] = Path(initial_params["weights"]).name
        if "data" in initial_params:
            initial_params["data"] = Path(initial_params["data"]).name

        self.log_params(initial_params)

    def _sanitize_dictionary(
        self, dictionary: Dict[Any, Any], target_type: type = str
    ) -> dict:
        """
        Sanitize a data dictionary to be compliant with MLflow metrics and params input.
        Invalid characters are replaced by an underscore (_).

        Keys can only contain alphanumerics, underscores (_), dashes (-), periods (.),
            spaces ( ), and slashes (/).
        Values must cast to *target_type*. If the cast fails, the pair will be ignored.

        Args:
            metrics (Dict[Any, Any]): A dictionary of metrics to sanitize.
            target_type (type, optional): The data type to cast the dictionary's Value to.
                Defaults to str.

        Examples:
            >>> metrics = {"x": 1, "y": 4.5}
            >>> sanitized_metrics = sanitize_metrics(metrics, float)
        """
        sanitized_dict = {}

        for k, v in dictionary.items():
            if not k:
                continue

            try:
                new_key = re.sub(REGEX_METRICS, "_", str(k))
                sanitized_dict[new_key] = target_type(v)
            except (ValueError, TypeError):
                continue

        return sanitized_dict

    def finish_run(self):
        """
        End the MLflow run.
        """
        if self.mlflow and self.mlflow.active_run():
            self.mlflow.end_run()

    def log_artifact(self, filename: str, artifact_path: str = "", epoch: int = -1):
        """
        Log an artifact as part of the current MLflow run.

        Args:
            filename (str): The local filename of the artifact.
            artifact_path (str, optional): The artifact logical path name.
                Defaults to the root folder.
            epoch (int, optional): Iteration number for the model.
                If greater than -1, an extra folder called epoch_*epoch*
                will be added to the logical path.
                Defaults to -1.

        Note:
            Azure ML will return an exception when trying to overwrite
            an existing artifact. The full logical path must be unique within the run.

        Examples:
            >>> mlflow_logger.log_artifact("/home/u/plot.png", "plots")
            >>> # Saved to plots/plot.png
            >>> mlflow_logger.log_artifact("/home/s/last.pt", "models", 2)
            >>> # Saved to models/epoch_2/last.pt
        """
        path = f"{artifact_path}/epoch_{epoch}" if epoch > -1 else artifact_path
        self.mlflow.log_artifact(filename, path)

    def log_debug_samples(self, files: List[Path], artifact_path: str):
        """
        Log a list of files split by iteration number. Files are logged
        as artifacts as part of the current MLflow run.

        Args:
            files (List[pathlib.Path]): A list of files to log.
            artifact_path (str): The artifact logical path name.
        """
        for file in files:
            if file.exists():
                it = re.search(r"_batch(\d+)", file.name)
                iteration = int(it.groups()[0]) if it else 0

                self.log_artifact(str(file), f"{artifact_path}/iteration_{iteration}")

    def log_metrics(self, metrics: Dict[str, float], epoch: int = 0):
        """
        Log a batch of metrics as part of the current MLflow run.

        Args:
            metrics (Dict[str, float]): A dictionary of metrics.
            epoch (int, optional): Iteration number for the batch of metrics.
                Defaults to 0.

        Note:
            Metric names can only contain alphanumerics, underscores (_), dashes (-),
                periods (.), spaces ( ), and slashes (/).
            Before logging, the names will be sanitized and any invalid character
                will be replaced.

        Examples:
            >>> metrics = {"x": 1, "y": 4.5}
            >>> mlflow_logger.log_metrics(metrics)
        """
        try:
            sanitized_metrics = self._sanitize_dictionary(metrics, float)
            self.mlflow.log_metrics(sanitized_metrics, step=epoch)
        except (TypeError, AttributeError, RestException) as e:
            logger.error(
                "%sFailed to log metrics %s for epoch %i: %s",
                PREFIX,
                metrics,
                epoch,
                str(e),
            )

    def log_params(self, params: Dict[str, Any]):
        """
        Log a batch of parameters as part of the current MLflow run.

        Args:
            params (Dict[str, Any]): A dictionary of parameters.

        Note:
            Parameter names can only contain alphanumerics, underscores (_), dashes (-),
                periods (.), spaces ( ), and slashes (/).
            If the parameter already exists with a different value,
            an MlflowException exception will be raised (INVALID_PARAMETER_VALUE).

        Examples:
            >>> params = {"x": "val", "y": 5.6}
            >>> mlflow_logger.log_params(params)
        """
        try:
            sanitized_params = self._sanitize_dictionary(params, str)
            self.mlflow.log_params(sanitized_params)
        except (TypeError, AttributeError, RestException) as e:
            logger.error(
                "%sFailed to log parameters %s: %s",
                PREFIX,
                params,
                str(e),
            )
