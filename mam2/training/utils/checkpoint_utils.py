# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import contextlib
import fnmatch
import logging
from typing import (
    Any,
    Callable,
    Dict,
    List,
    Optional,
    Sequence,
    Set,
    Union,
)

import numpy as np
import torch
import torch.nn as nn
from iopath.common.file_io import g_pathmgr


def unix_pattern_to_parameter_names(
    constraints: List[str], all_parameter_names: Sequence[str]
) -> Union[None, Set[str]]:
    """
    Go through the list of parameter names and select those that match
    any of the provided constraints
    """
    parameter_names = []
    for param_name in constraints:
        matching_parameters = set(fnmatch.filter(all_parameter_names, param_name))
        assert len(matching_parameters) > 0, (
            f"param_names {param_name} don't match any param in the given names."
        )
        parameter_names.append(matching_parameters)
    return set.union(*parameter_names)


def filter_params_matching_unix_pattern(
    patterns: List[str], state_dict: Dict[str, torch.Tensor]
) -> Dict[str, torch.Tensor]:
    """
    Remove from the state dictionary the parameters matching the provided unix patterns

    Args:
        patterns: the list of unix patterns to exclude
        state_dict: the dictionary to filter

    Returns:
        A new state dictionary
    """
    if len(patterns) == 0:
        return {}

    all_keys = list(state_dict.keys())
    included_keys = unix_pattern_to_parameter_names(patterns, all_keys)
    return {k: state_dict[k] for k in included_keys}


def exclude_params_matching_unix_pattern(
    patterns: List[str], state_dict: Dict[str, torch.Tensor]
) -> Dict[str, torch.Tensor]:
    """
    Remove from the state dictionary the parameters matching the provided unix patterns

    Args:
        patterns: the list of unix patterns to exclude
        state_dict: the dictionary to filter

    Returns:
        A new state dictionary
    """
    if len(patterns) == 0:
        return state_dict

    all_keys = list(state_dict.keys())
    excluded_keys = unix_pattern_to_parameter_names(patterns, all_keys)
    return {k: v for k, v in state_dict.items() if k not in excluded_keys}


def _get_state_dict_summary(state_dict: Dict[str, torch.Tensor]):
    keys = []
    trace = []
    for k, v in state_dict.items():
        keys.append(k)
        trace.append(v.sum().item())
    trace = np.array(trace)[np.argsort(keys)]
    return trace


def assert_skipped_parameters_are_frozen(model: nn.Module, patterns: List[str]):
    """
    Verifies that all the parameters matching the provided patterns
    are frozen - this acts as a safeguard when ignoring parameter
    when saving checkpoints - if the parameters are in fact trainable
    """
    if not patterns:
        return

    frozen_state_dict = filter_params_matching_unix_pattern(
        patterns=patterns, state_dict=model.state_dict()
    )
    non_frozen_keys = {
        n for n, p in model.named_parameters() if n in frozen_state_dict and p.requires_grad
    }
    if non_frozen_keys:
        raise ValueError(
            f"Parameters excluded with `skip_saving_parameters` should be frozen: {non_frozen_keys}"
        )


@contextlib.contextmanager
def with_check_parameter_frozen(model: nn.Module, patterns: List[str], disabled: bool = True):
    """
    Context manager that inspects a model surrounding a piece of code
    and verifies if the model has been updated by this piece of code

    The function will raise an exception if the model has been updated
    on at least one of the parameter that matches one of the pattern

    Args:
        model: the model that might have been updated
        patterns: for the parameters we want to observe
        allowed:
    """
    if not patterns or disabled:
        yield
        return

    frozen_state_dict = filter_params_matching_unix_pattern(
        patterns=patterns, state_dict=model.state_dict()
    )
    summary_before = _get_state_dict_summary(frozen_state_dict)

    yield

    frozen_state_dict = filter_params_matching_unix_pattern(
        patterns=patterns, state_dict=model.state_dict()
    )
    summary_after = _get_state_dict_summary(frozen_state_dict)

    if not np.allclose(summary_before, summary_after, atol=1e-6):
        raise ValueError(
            """
            The `model_weight_initializer` has initialized parameters frozen with `skip_saving_parameters`.
            You can resolve this error by either initializing those parameters from within the model definition
            or using the flag `trainer.checkpoint.initialize_after_preemption` to True.
        """
        )


def load_checkpoint(
    path_list: List[str],
    pick_recursive_keys: Optional[List[str]] = None,
    map_location: str = "cpu",
) -> Any:
    """
    Loads a checkpoint from the specified path.

    Args:
        path_list: A list of paths which contain the checkpoint. Each element
            is tried (in order) until a file that exists is found. That file is then
            used to read the checkpoint.
        pick_recursive_keys: Picks sub dicts from the loaded checkpoint if not None.
            For pick_recursive_keys = ["a", "b"], will return checkpoint_dict["a"]["b"]
        map_location (str): a function, torch.device, string or a dict specifying how to
            remap storage locations

    Returns: Model with the matchin pre-trained weights loaded.
    """
    path_exists = False
    for path in path_list:
        if g_pathmgr.exists(path):
            path_exists = True
            break

    if not path_exists:
        raise ValueError(f"No path exists in {path_list}")

    with g_pathmgr.open(path, "rb") as f:
        checkpoint = torch.load(f, map_location=map_location)

    logging.info(f"Loaded checkpoint from {path}")
    if pick_recursive_keys is not None:
        for key in pick_recursive_keys:
            checkpoint = checkpoint[key]
    return checkpoint


def check_load_state_dict_errors(
    missing_keys,
    unexpected_keys,
    strict: bool,
    ignore_missing_keys: List[str] = None,
    ignore_unexpected_keys: List[str] = None,
):
    if ignore_missing_keys is not None and len(ignore_missing_keys) > 0:
        ignored_keys = unix_pattern_to_parameter_names(ignore_missing_keys, missing_keys)
        missing_keys = [key for key in missing_keys if key not in ignored_keys]

    if ignore_unexpected_keys is not None and len(ignore_unexpected_keys) > 0:
        ignored_unexpected_keys = unix_pattern_to_parameter_names(
            ignore_unexpected_keys, unexpected_keys
        )
        unexpected_keys = [key for key in unexpected_keys if key not in ignored_unexpected_keys]

    err = "State key mismatch."
    if unexpected_keys:
        err += f" Unexpected keys: {unexpected_keys}."
    if missing_keys:
        err += f" Missing keys: {missing_keys}."

    if unexpected_keys or missing_keys:
        logging.warning(err)
        if unexpected_keys or strict:
            raise KeyError(err)


def load_state_dict_into_model(
    state_dict: Dict,
    model: nn.Module,
    strict: bool = True,
    ignore_missing_keys: List[str] = None,
    ignore_unexpected_keys: List[str] = None,
    checkpoint_kernels: List[Callable] = None,
):
    """
    Loads a state dict into the given model.

    Args:
        state_dict: A dictionary containing the model's
            state dict, or a subset if strict is False
        model: Model to load the checkpoint weights into
        strict: raise if the state_dict has missing state keys
        ignore_missing_keys: unix pattern of keys to ignore
    """
    # Apply kernels
    if checkpoint_kernels is not None:
        for f in checkpoint_kernels:
            state_dict = f(state_dict=state_dict)
    from mam2.checkpoints import canonical_state_dict

    missing_keys, unexpected_keys = model.load_state_dict(
        canonical_state_dict(state_dict), strict=False
    )

    check_load_state_dict_errors(
        missing_keys,
        unexpected_keys,
        strict=strict,
        ignore_missing_keys=ignore_missing_keys,
        ignore_unexpected_keys=ignore_unexpected_keys,
    )
    return model
