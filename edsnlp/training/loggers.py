import csv
import json
import os
import warnings
from typing import Any, Dict, Optional, Union

import accelerate.tracking
from rich_logger import RichTablePrinter

import edsnlp


def flatten_dict(d, path=""):
    if not isinstance(d, (list, dict)):
        return {path: d}

    if isinstance(d, list):
        items = enumerate(d)
    else:
        items = d.items()

    return {
        k: v
        for key, val in items
        for k, v in flatten_dict(val, f"{path}/{key}" if path else key).items()
    }


class CSVTracker(accelerate.tracking.GeneralTracker):
    name = "csv"
    requires_logging_directory = True

    def __init__(
        self,
        *,
        logging_dir: Union[str, os.PathLike],
        file_name: str = "metrics.csv",
        **kwargs,
    ):
        super().__init__()
        self.logging_dir = logging_dir

        self.file_path = os.path.join(self.logging_dir, file_name)
        self._file = None
        self._writer = None
        self._columns = None
        self._has_header = False

    @accelerate.tracking.on_main_process
    def start(self):
        if self._file is not None:
            return
        os.makedirs(self.logging_dir, exist_ok=True)
        self._file = open(self.file_path, mode="w", newline="")
        self._writer = csv.writer(self._file)

    @property
    def tracker(self):  # pragma: no cover
        return None

    @accelerate.tracking.on_main_process
    def store_init_configuration(self, values: Dict[str, Any]):
        pass

    @accelerate.tracking.on_main_process
    def log(self, values: Dict[str, Any], step: Optional[int] = None):
        """
        Logs `values` to the CSV file, at an optional `step`.

        - If it's the first call, the columns are inferred from the keys in `values`
          plus a "step" column if the user provides `step`.
        - All subsequent calls must use the same columns. Any missing columns get
          written as empty, any new columns generate a warning.
        """
        values = flatten_dict(values)

        if self._columns is None:
            self._columns = list({**{"step": None}, **values}.keys())
            self._writer.writerow(self._columns)
            self._has_header = True

        # Build a row in the order of self._columns
        row = []
        for col in self._columns:
            if col == "step":
                row.append(step if step is not None else "")
            else:
                if col not in values and col != "step":
                    row.append("")
                else:
                    row.append(values.get(col, ""))

        for extra_key in values.keys():
            if extra_key not in self._columns:
                warnings.warn(
                    f"CSVTracker: encountered a new field '{extra_key}' that was not in"
                    f"the field keys of the first logged step. It will not be logged."
                )

        self._writer.writerow(row)
        self._file.flush()

    @accelerate.tracking.on_main_process
    def finish(self):
        self._file.close()


class JSONTracker(accelerate.tracking.GeneralTracker):
    name = "json"
    requires_logging_directory = True

    def __init__(
        self,
        logging_dir: Union[str, os.PathLike],
        file_name: str = "metrics.json",
        **kwargs,
    ):
        super().__init__()
        self.logging_dir = logging_dir
        self.initialized = False

        self._file_path = os.path.join(self.logging_dir, file_name)
        self._logs = []

    @property
    def tracker(self):  # pragma: no cover
        return None

    @accelerate.tracking.on_main_process
    def store_init_configuration(self, values: Dict[str, Any]):
        pass

    @accelerate.tracking.on_main_process
    def log(self, values: Dict[str, Any], step: Optional[int] = None):
        """
        Logs `values` along with a `step` (if provided).

        On every call, we:
          1. Append a new record to our in-memory list.
          2. Write out the entire JSON file containing all records.
        """
        log_entry = {"step": step, **values}
        self._logs.append(log_entry)
        os.makedirs(self.logging_dir, exist_ok=True)

        with open(self._file_path, mode="w") as f:
            json.dump(self._logs, f, indent=2)

    @accelerate.tracking.on_main_process
    def finish(self):
        pass


class RichTracker(accelerate.tracking.GeneralTracker):
    DEFAULT_FIELDS = {
        "step": {},
        "per_pipe/(.*)/results/loss": {
            "goal": "lower_is_better",
            "format": "{:.2e}",
            "goal_wait": 2,
            "name": r"\1_loss",
        },
        "lr": {"format": "{:.2e}"},
        "validation/speed/(.*)": {"format": "{:.2f}", r"name": r"\1"},
        "validation/(.*?)/micro/(f|r|p)$": {
            "goal": "higher_is_better",
            "format": "{:.2%}",
            "goal_wait": 1,
            "name": r"\1_\2",
        },
        "validation/(.*?)/(uas|las)": {
            "goal": "higher_is_better",
            "format": "{:.2%}",
            "goal_wait": 1,
            "name": r"\1_\2",
        },
        "weights/grad_norm/__all__": {
            "format": "{:.2e}",
            "name": "grad_norm",
        },
    }

    name = "rich"
    requires_logging_directory = False

    def __init__(
        self,
        run_name: Optional[str] = None,
        fields: Dict[str, Union[Dict, bool]] = None,
        key: Optional[str] = None,
        hijack_tqdm: bool = True,
        **kwargs,
    ):
        super().__init__()

        self.run_name = run_name
        fields = fields if fields is not None else self.DEFAULT_FIELDS
        self.fields = fields or {}
        self.hijack_tqdm = hijack_tqdm
        self.printer = None
        self.key = key

    @accelerate.tracking.on_main_process
    def start(self):
        self.printer = RichTablePrinter(key=self.key, fields=self.fields)

        if self.hijack_tqdm:
            self.printer.hijack_tqdm()

    @property
    def tracker(self):
        return self.printer

    @accelerate.tracking.on_main_process
    def store_init_configuration(self, values: Dict[str, Any]):
        pass

    @accelerate.tracking.on_main_process
    def log(self, values: Dict[str, Any], step: Optional[int] = None):
        """
        Logs values in the Rich table. If `step` is provided, we include it in the
        logged data.
        """
        combined = {"step": step, **flatten_dict(values)}
        self.printer.log_metrics(combined)

    @accelerate.tracking.on_main_process
    def finish(self):
        """
        Finalize the table (e.g., stop rendering).
        """
        self.printer.finalize()


class TensorBoardTracker(accelerate.tracking.TensorBoardTracker):
    def __init__(
        self,
        project_name: str,
        logging_dir: Optional[Union[str, os.PathLike]] = None,
    ):
        env_logging_dir = os.environ.get("TENSORBOARD_LOGGING_DIR", None)
        if env_logging_dir is not None and logging_dir is not None:  # pragma: no cover
            warnings.warn(
                f"Using the env TENSORBOARD_LOGGING_DIR={env_logging_dir} as the "
                f"logging directory for TensorBoard, instead of ${logging_dir}."
            )
            logging_dir = env_logging_dir
        assert logging_dir is not None, (
            "Please provide a logging directory or set TENSORBOARD_LOGGING_DIR"
        )
        super().__init__(project_name, logging_dir)

    def store_init_configuration(self, values: Dict[str, Any]):
        values = flatten_dict(values)
        return super().store_init_configuration(values)

    def log(self, values: dict, step: Optional[int] = None, **kwargs):
        values = flatten_dict(values)
        return super().log(values, step, **kwargs)


class AimTracker(accelerate.tracking.AimTracker):
    main_process_only = True

    def __init__(
        self,
        project_name: str,
        logging_dir: Optional[Union[str, os.PathLike]] = None,
        # We set it to None by default, as it's rarely useful
        system_tracking_interval: Optional[int] = None,
        **kwargs,
    ):
        env_logging_dir = os.environ.get("AIM_LOGGING_DIR", None)
        if env_logging_dir is not None and logging_dir is not None:
            warnings.warn(
                f"Using the env AIM_LOGGING_DIR={env_logging_dir} as the logging "
                f"directory for Aim, instead of ${logging_dir}."
            )
            logging_dir = env_logging_dir
        assert logging_dir is not None, (
            "Please provide a logging directory or set AIM_LOGGING_DIR"
        )

        super().__init__(
            run_name=project_name,
            logging_dir=logging_dir,
            system_tracking_interval=system_tracking_interval,
            **kwargs,
        )

    @accelerate.tracking.on_main_process
    def start(self):
        from aim import Run

        self.writer = Run(
            repo=self.aim_repo_path, experiment=self.run_name, **self.init_kwargs
        )
        accelerate.tracking.logger.debug(
            f"Initialized Aim run {self.writer.hash} in project {self.run_name}"
        )

    def log(self, values: dict, step: Optional[int], **kwargs):
        values = flatten_dict(values)
        return super().log(values, step, **kwargs)


@edsnlp.registry.loggers.register("json")
def JSONLogger(
    logging_dir: Union[str, os.PathLike],
    file_name: str = "metrics.json",
    **kwargs,
) -> JSONTracker:  # pragma: no cover
    """
    A simple JSON-based logger that writes logs to a JSON file as a
    list of dictionaries. By default, with `edsnlp.train` the JSON file
    is located under a local directory `${CWD}/artifact/metrics.json`.

    This method is not recommended for large and frequent logging, as it
    re-writes the entire JSON file on every call. Prefer
    [`CSVLogger`][edsnlp.training.loggers.CSVLogger] for frequent
    and heavy logging.

    Parameters
    ----------
    logging_dir : str or os.PathLike
        Directory in which to store the JSON file.
    file_name : str, optional
        Name of the JSON file. Defaults to "metrics.json".
    """
    return JSONTracker(
        logging_dir=logging_dir,
        file_name=file_name,
        **kwargs,
    )


@edsnlp.registry.loggers.register("csv")
def CSVLogger(
    logging_dir: Union[str, os.PathLike],
    file_name: str = "metrics.csv",
    **kwargs,
) -> CSVTracker:  # pragma: no cover
    """
    A simple CSV-based logger that writes logs to a CSV file. By default,
    with `edsnlp.train` the CSV file is located under a local directory
    `${CWD}/artifact/metrics.csv`.

    !!! warning "Consistent Keys"

        This logger expects that the `values` dictionary passed to `log` has
        consistent keys across all calls. If a new key is encountered in a
        subsequent call, it will be ignored and a warning will be issued.

    Parameters
    ----------
    logging_dir : str or os.PathLike
      Directory in which to store the CSV.
    file_name : str, optional
      Name of the CSV file. Defaults to "metrics.csv".
    """
    return CSVTracker(
        logging_dir=logging_dir,
        file_name=file_name,
        **kwargs,
    )


@edsnlp.registry.loggers.register("rich")
def RichLogger(
    run_name: Optional[str] = None,
    fields: Dict[str, Union[Dict, bool]] = None,
    key: Optional[str] = None,
    hijack_tqdm: bool = True,
    **kwargs,
) -> RichTracker:  # pragma: no cover
    """
    A logger that displays logs in a Rich-based table using
    [rich-logger](https://github.com/percevalw/rich-logger).
    This logger is also available via the loggers registry as `rich`.

    !!! warning "No Disk Logging"

        This logger doesn't save logs to disk. It's meant for displaying
        logs in a pretty table during training. If you need to save logs
        to disk, consider combining this logger with any other logger.

    Parameters
    ----------
    fields: Dict[str, Union[Dict, bool]]
        Field descriptors containing goal ("lower_is_better" or "higher_is_better"),
         format and display name
        The key is a regex that will be used to match the fields to log
        Each entry of the dictionary should match the following scheme:

        - key: a regex to match columns
        - value: either a Dict or False to hide the column, the dict format is
            - name: the name of the column
            - goal: "lower_is_better" or "higher_is_better"

        This defaults to a set of metrics and stats that are commonly
        logged during EDS-NLP training.
    key: Optional[str]
        Key to group the logs
    hijack_tqdm: bool
        Whether to replace the tqdm progress bar with a rich progress bar.
        Indeed, rich progress bars integrate better with the rich table.
    """
    return RichTracker(
        run_name=run_name,
        fields=fields,
        key=key,
        hijack_tqdm=hijack_tqdm,
        **kwargs,
    )


@edsnlp.registry.loggers.register("aim")
def AimLogger(
    project_name: str,
    logging_dir: Optional[Union[str, os.PathLike]] = None,
    **kwargs,
) -> accelerate.tracking.AimTracker:  # pragma: no cover
    """
    Logger for [Aim](https://github.com/aimhubio/aim).

    Parameters
    ----------
    project_name: str
        Name of the project.
    logging_dir: Optional[Union[str, os.PathLike]]
        Directory in which to store the Aim logs.
        The environment variable `AIM_LOGGING_DIR` takes precedence over this
        argument.
    kwargs: Dict
        Additional keyword arguments to pass to the Aim init function.
    """
    return AimTracker(
        project_name,
        logging_dir=logging_dir,
        **kwargs,
    )


@edsnlp.registry.loggers.register("tensorboard")
def TensorBoardLogger(
    project_name: str,
    logging_dir: Optional[Union[str, os.PathLike]] = None,
) -> accelerate.tracking.TensorBoardTracker:  # pragma: no cover
    """
    Logger for [TensorBoard](https://github.com/tensorflow/tensorboard).
    This logger is also available via the loggers registry as `tensorboard`.

    Parameters
    ----------
    project_name: str
        Name of the project.
    logging_dir: Union[str, os.PathLike]
        Directory in which to store the TensorBoard logs. Logs of different runs
        will be stored in `logging_dir/project_name`.
        The environment variable `TENSORBOARD_LOGGING_DIR` takes precedence over
        this argument.
    """
    return TensorBoardTracker(
        project_name,
        logging_dir=logging_dir,
    )


@edsnlp.registry.loggers.register("wandb")
def WandBLogger(
    project_name: str,
    **kwargs,
) -> accelerate.tracking.WandBTracker:  # pragma: no cover
    """
    Logger for [Weights & Biases](https://docs.wandb.ai/quickstart/).
    This logger is also available via the loggers registry as `wandb`.

    Parameters
    ----------
    project_name: str
        Name of the project. This will become the `project`
        parameter in `wandb.init`.
    kwargs: Dict
        Additional keyword arguments to pass to the WandB init function.

    Returns
    -------
    accelerate.tracking.WandBTracker
    """
    return accelerate.tracking.WandBTracker(project_name, **kwargs)


@edsnlp.registry.loggers.register("mlflow")
def MLflowLogger(
    project_name: str,
    logging_dir: Optional[Union[str, os.PathLike]] = None,
    run_id: Optional[str] = None,
    tags: Optional[Union[Dict[str, Any], str]] = None,
    nested_run: Optional[bool] = False,
    run_name: Optional[str] = None,
    description: Optional[str] = None,
) -> accelerate.tracking.MLflowTracker:  # pragma: no cover
    """
    Logger for
    [MLflow](https://mlflow.org/docs/latest/getting-started/intro-quickstart/).
    This logger is also available via the loggers registry as `mlflow`.

    Parameters
    ----------
    project_name: str
        Name of the project. This will become the mlflow experiment name.
    logging_dir: Optional[Union[str, os.PathLike]]
        Directory in which to store the MLflow logs.
    run_id: Optional[str]
        If specified, get the run with the specified UUID and log parameters and metrics
        under that run. The run’s end time is unset and its status is set to running,
        but the run’s other attributes (source_version, source_type, etc.) are not
        changed. Environment variable MLFLOW_RUN_ID has priority over this argument.
    tags: Optional[Union[Dict[str, Any], str]]
        An optional `dict` of `str` keys and values, or a `str` dump from a `dict`, to
        set as tags on the run. If a run is being resumed, these tags are set on the
        resumed run. If a new run is being created, these tags are set on the new run.
        Environment variable MLFLOW_TAGS has priority over this argument.
    nested_run: Optional[bool]
        Controls whether run is nested in parent run. True creates a nested run.
        Environment variable MLFLOW_NESTED_RUN has priority over this argument.
    run_name: Optional[str]
        Name of new run (stored as a mlflow.runName tag). Used only when `run_id` is
        unspecified.
    description: Optional[str]
        An optional string that populates the description box of the run. If a run is
        being resumed, the description is set on the resumed run. If a new run is being
        created, the description is set on the new run.

    Returns
    -------
    accelerate.tracking.MLflowTracker
    """
    return accelerate.tracking.MLflowTracker(
        project_name,
        logging_dir=logging_dir,
        run_id=run_id,
        tags=tags,
        nested_run=nested_run,
        run_name=run_name,
        description=description,
    )


@edsnlp.registry.loggers.register("cometml")
def CometMLLogger(
    project_name: str,
    **kwargs,
) -> accelerate.tracking.CometMLTracker:  # pragma: no cover
    """
    Logger for [CometML](https://www.comet.com/docs/).
    This logger is also available via the loggers registry as `cometml`.

    Parameters
    ----------
    project_name: str
        Name of the project.
    kwargs: Dict
        Additional keyword arguments to pass to the CometML Experiment
        object.

    Returns
    -------
    accelerate.tracking.CometMLTracker
    """
    return accelerate.tracking.CometMLTracker(project_name, **kwargs)


try:
    from accelerate.tracking import ClearMLTracker as _ClearMLTracker

    @edsnlp.registry.loggers.register("clearml")
    def ClearMLLogger(
        project_name: str,
        **kwargs,
    ) -> accelerate.tracking.ClearMLTracker:  # pragma: no cover
        """
        Logger for
        [ClearML](https://clear.ml/docs/latest/docs/getting_started/ds/ds_first_steps/).
        This logger is also available via the loggers registry as `clearml`.

        Parameters
        ----------
        project_name: str
            Name of the experiment. Environment variables `CLEARML_PROJECT` and
            `CLEARML_TASK` have priority over this argument.
        kwargs: Dict
            Additional keyword arguments to pass to the ClearML Task object.

        Returns
        -------
        accelerate.tracking.ClearMLTracker
        """
        return _ClearMLTracker(project_name, **kwargs)
except ImportError:  # pragma: no cover

    def ClearMLLogger(*args, **kwargs):
        raise ImportError("ClearMLLogger is not available.")


try:
    from accelerate.tracking import DVCLiveTracker as _DVCLiveTracker

    @edsnlp.registry.loggers.register("dvclive")
    def DVCLiveLogger(
        live: Any = None,
        **kwargs,
    ) -> accelerate.tracking.DVCLiveTracker:  # pragma: no cover
        """
        Logger for [DVC Live](https://dvc.org/doc/dvclive).
        This logger is also available via the loggers registry as `dvclive`.

        Parameters
        ----------
        live: dvclive.Live
            An instance of `dvclive.Live` to use for logging.
        kwargs: Dict
            Additional keyword arguments to pass to the `dvclive.Live` constructor.

        Returns
        -------
        accelerate.tracking.DVCLiveTracker
        """
        return _DVCLiveTracker(None, live=live, **kwargs)
except ImportError:  # pragma: no cover

    def DVCLiveLogger(*args, **kwargs):
        raise ImportError("DVCLiveLogger is not available.")
