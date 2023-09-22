import ast
import importlib
import inspect
import logging
import sys
from typing import Union

import astunparse
from griffe import Extension, Object, ObjectNode
from griffe.docstrings.dataclasses import DocstringSectionParameters
from griffe.expressions import Expr
from griffe.logger import patch_loggers


def get_logger(name):
    new_logger = logging.getLogger(name)
    new_logger.setLevel("ERROR")
    return new_logger


patch_loggers(get_logger)

logger = get_logger(__name__)


class EDSNLPDocstrings(Extension):
    def __init__(self):
        super().__init__()

        self.PIPE_OBJ = {}
        self.FACT_MEM = {}
        self.PIPE_TO_FACT = {}

    def on_instance(self, node: Union[ast.AST, ObjectNode], obj: Object) -> None:
        if (
            isinstance(node, ast.Assign)
            and obj.name == "create_component"
            and isinstance(node.value, ast.Call)
            and isinstance(node.value.func, ast.Call)
        ):

            module_name = obj.path.rsplit(".", 1)[0]
            for name, mod in list(sys.modules.items()):
                if name.startswith("edspdf"):
                    importlib.reload(mod)
            module = importlib.reload(importlib.import_module(module_name))

            config_node = node.value.func
            config_node = next(
                (kw.value for kw in config_node.keywords if kw.arg == "default_config"),
                None,
            )
            try:
                default_config = eval(astunparse.unparse(config_node), module.__dict__)
            except Exception:
                default_config = {}

            # import object to get its evaluated docstring
            try:
                runtime_obj = getattr(module, obj.name)
                source = inspect.getsource(runtime_obj)
                self.visit(ast.parse(source))
            except ImportError:
                logger.debug(f"Could not get dynamic docstring for {obj.path}")
                return
            except AttributeError:
                logger.debug(f"Object {obj.path} does not have a __doc__ attribute")
                return

            spec = inspect.getfullargspec(runtime_obj)
            func_defaults = dict(
                zip(spec.args[-len(spec.defaults) :], spec.defaults)
                if spec.defaults
                else (),
                **(spec.kwonlydefaults or {}),
            )
            defaults = {**func_defaults, **default_config}
            self.FACT_MEM[obj.path] = (node, obj, defaults)
            pipe_path = runtime_obj.__module__ + "." + runtime_obj.__name__
            self.PIPE_TO_FACT[pipe_path] = obj.path

            if pipe_path in self.PIPE_OBJ:
                pipe = self.PIPE_OBJ[pipe_path]
                obj.docstring = pipe.docstring
            else:
                return
        elif obj.is_class or obj.is_function:
            self.PIPE_OBJ[obj.path] = obj
            if obj.path in self.PIPE_TO_FACT:
                node, fact_obj, defaults = self.FACT_MEM[self.PIPE_TO_FACT[obj.path]]
                fact_obj.docstring = obj.docstring
                obj = fact_obj
            else:
                return
        else:
            return

        if obj.docstring is None:
            return

        param_section: DocstringSectionParameters = None
        obj.docstring.parser = "numpy"
        for section in obj.docstring.parsed:
            if isinstance(section, DocstringSectionParameters):
                param_section = section  # type: ignore

        if param_section is None:
            return

        for param in param_section.value:
            if param.name in defaults:
                param.default = str(defaults[param.name])
            if isinstance(param.default, Expr):
                continue
            if param.default is not None and len(param.default) > 50:
                param.default = param.default[: 50 - 3] + "..."
