from copy import deepcopy
from typing import Any, Dict, Union

from thinc.config import Config


def merge_configs(
    config: Union[Dict[str, Any], Config],
    *updates: Union[Dict[str, Any], Config],
    remove_extra: bool = False,
) -> Union[Dict[str, Any], Config]:
    """Deep merge two configs."""

    def deep_set(current, path, val):
        path = path.split(".")
        for part in path[:-1]:
            current = current[part]
        current[path[-1]] = val

    def rec(old, new):
        if remove_extra:
            # Filter out values in the original config that are not in defaults
            keys = list(new.keys())
            for key in keys:
                if key not in old:
                    del new[key]
        for key, new_val in list(new.items()):
            if "." in key:
                deep_set(old, key, new_val)
                continue

            if key not in old:
                old[key] = new_val
                continue

            old_val = old[key]
            if isinstance(old_val, dict) and isinstance(new_val, dict):
                old_promise = next((k for k in old_val if k.startswith("@")), None)
                new_promise = next((k for k in new_val if k.startswith("@")), None)
                if (
                    new_promise is not None
                    and old_promise != new_promise
                    or old_val.get(old_promise) != new_val.get(new_promise)
                ):
                    old[key] = new_val
                else:
                    rec(old[key], new_val)
            else:
                old[key] = new_val
        return old

    config = deepcopy(config)
    for u in updates:
        u = deepcopy(u)
        rec(config, u)
    return Config(config)
