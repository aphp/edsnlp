# flake8: noqa: F811
import ast
import importlib
import inspect
import os


def lazify():
    def _get_module_paths(file):
        """
        Reads the content of the current file, parses it with ast and store the
        import path for future potential imports. This is useful to only import
        the module that is requested and avoid loading all the modules at once, since
        some of them are quite heavy, or contain dependencies that are not always
        available.

        For instance:
        > from .trainable.span_qualifier.factory import create_component as
        span_qualifier is stored in the cache as:
        > module_paths["span_qualifier"] = "trainable.span_qualifier.factory"

        Returns
        -------
        Dict[str, Tuple[str, str]]
            The absolute path of the current file.
        """
        module_path = os.path.abspath(file)
        with open(module_path, "r") as f:
            module_content = f.read()
        module_ast = ast.parse(module_content)
        module_paths = {}
        for node in module_ast.body:
            # Lookup TYPE_CHECKING
            if not (
                isinstance(node, ast.If)
                and (
                    (
                        isinstance(node.test, ast.Name)
                        and node.test.id == "TYPE_CHECKING"
                    )
                    or (
                        isinstance(node.test, ast.Attribute)
                        and node.test.attr == "TYPE_CHECKING"
                    )
                )
            ):
                continue
            for import_node in node.body:
                if isinstance(import_node, ast.ImportFrom):
                    for name in import_node.names:
                        module_paths[name.asname or name.name] = (
                            import_node.module,
                            name.name,
                        )

        return module_paths

    def __getattr__(name):
        """
        Imports the actual module if it is in the module_paths dict.

        Parameters
        ----------
        name

        Returns
        -------

        """
        if name in module_paths:
            module_path, module_name = module_paths[name]
            result = getattr(
                importlib.__import__(
                    module_path,
                    fromlist=[module_name],
                    globals=module_globals,
                    level=1,
                ),
                module_name,
            )
            module_globals[name] = result
            return result
        raise AttributeError(f"module {__name__} has no attribute {name}")

    def __dir__():
        """
        Returns the list of available modules.

        Returns
        -------
        List[str]
        """
        return __all__

    # Access upper frame
    module_globals = inspect.currentframe().f_back.f_globals

    module_paths = _get_module_paths(module_globals["__file__"])

    __all__ = list(module_paths.keys())

    module_globals.update(
        {
            "__getattr__": __getattr__,
            "__dir__": __dir__,
            "__all__": __all__,
        }
    )
