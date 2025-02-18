# Loggers

When training a model, it is important to keep track of the training process, model performance at different stages, and statistics about the training data over time. This is where loggers come in. Loggers are used to store such information to be able to analyze and visualize it later.

The EDS-NLP training API (`edsnlp.train`) relies on `accelerate` integration of popular loggers, as well as a few custom loggers.
You can configure loggers in `edsnlp.train` via the `logger` parameter of the `train` function by specifying:

- a string or a class instance or partially initialized class instance of a logger, e.g.

    === "Via the Python API"
        ```{ .python .no-check }
        from edsnlp.training.loggers import CSVLogger
        from edsnlp.training import train

        logger = CSVLogger.draft()
        train(..., logger=logger)
        # or train(..., logger="csv")
        ```

    === "Via a config file"
        ```yaml
        train:
          ...
          logger:
            "@loggers": csv
            ...
        ```


- or a list of string / logger instances, e.g.

    === "Via the Python API"
        ```{ .python .no-check }
        from edsnlp.training.loggers import CSVLogger
        from edsnlp.training import train

        loggers = ["tensorboard", CSVLogger.draft(...)]
        train(..., logger=loggers)
        ```

    === "Via a config file"
        ```yaml
        train:
          ...
          logger:
              - tensorboard  # as a string
              - "@loggers": csv  # as a (partially) instanciated logger
                ...
        ```

!!! note "Draft objects"

    `edsnlp.train` will provide a default project name and logging dir for loggers that require these parameters, but it is
recommended to set the project name explicitly in the logger configuration. For these loggers, if you don't want to set
the project name yourself, you can either:

    - call `CSVLogger.draft(...)` without the normal init parameters minus the `project_name` or `logging_dir` parameters,
      which will cause a `Draft[CSVLogger]` object to be returned if some required parameters are missing
    - or use `"@loggers": csv` in the config file, which will also cause a `Draft[CSVLogger]` object to be returned if some required
      parameters are missing

    If you do not want a `Draft` object to be returned, call `CSVLogger` directly.

The supported loggers are listed below.

### RichLogger {: #edsnlp.training.loggers.RichLogger }

::: edsnlp.training.loggers.RichLogger.__init__
    options:
        sections: ["text", "parameters"]
        heading_level: 4
        show_bases: false
        show_source: false
        only_class_level: true

### CSVLogger {: #edsnlp.training.loggers.CSVLogger }

::: edsnlp.training.loggers.CSVLogger.__init__
    options:
        sections: ["text", "parameters"]
        heading_level: 4
        show_bases: false
        show_source: false
        only_class_level: true

### JSONLogger {: #edsnlp.training.loggers.JSONLogger }

::: edsnlp.training.loggers.JSONLogger.__init__
    options:
        sections: ["text", "parameters"]
        heading_level: 4
        show_bases: false
        show_source: false
        only_class_level: true

### TensorBoardLogger {: #edsnlp.training.loggers.TensorBoardLogger }

::: edsnlp.training.loggers.TensorBoardLogger
    options:
        sections: ["text", "parameters"]
        heading_level: 4
        show_bases: false
        show_source: false
        only_class_level: true

### AimLogger {: #edsnlp.training.loggers.AimLogger }

::: edsnlp.training.loggers.AimLogger
    options:
        sections: ["text", "parameters"]
        heading_level: 4
        show_bases: false
        show_source: false
        only_class_level: true

### WandBLogger {: #edsnlp.training.loggers.WandBLogger }

::: edsnlp.training.loggers.WandBLogger
    options:
        sections: ["text", "parameters"]
        heading_level: 4
        show_bases: false
        show_source: false
        only_class_level: true

### MLflowLogger {: #edsnlp.training.loggers.MLflowLogger }

::: edsnlp.training.loggers.MLflowLogger
    options:
        sections: ["text", "parameters"]
        heading_level: 4
        show_bases: false
        show_source: false
        only_class_level: true

### CometMLLogger {: #edsnlp.training.loggers.CometMLLogger }

::: edsnlp.training.loggers.CometMLLogger
    options:
        sections: ["text", "parameters"]
        heading_level: 4
        show_bases: false
        show_source: false
        only_class_level: true

### DVCLiveLogger {: #edsnlp.training.loggers.DVCLiveLogger }

::: edsnlp.training.loggers.DVCLiveLogger
    options:
        sections: ["text", "parameters"]
        heading_level: 4
        show_bases: false
        show_source: false
        only_class_level: true
