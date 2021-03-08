# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.

from collections import defaultdict
from typing import Any, Dict, Iterable, List, Optional, Set, Tuple

import tabulate
import torch
from torch import nn

from .activation_count import ActivationCountAnalysis
from .flop_count import FlopCountAnalysis
from .parameter_count import parameter_count


### Pre-processing functions ###


def _format_size(x: int, sig_figs: int = 3, hide_zero: bool = False) -> str:
    """
    Formats an integer for printing in a table or model representation.
    Expresses the number in terms of 'kilo', 'mega', etc., using
    'K', 'M', etc. as a suffix.

    Args:
        x (int) : The integer to format.
        sig_figs (int) : The number of significant figures to keep
        hide_zero (bool) : If True, x=0 is replaced with an empty string
            instead of '0'.

    Returns:
        str : The formatted string.
    """
    if hide_zero and x == 0:
        return str("")
    frmt = "{{:.{}}}".format(sig_figs)
    if abs(x) > 1e14:
        return (frmt + "P").format(x / 1e15)
    if abs(x) > 1e11:
        return (frmt + "T").format(x / 1e12)
    if abs(x) > 1e8:
        return (frmt + "G").format(x / 1e9)
    if abs(x) > 1e5:
        return (frmt + "M").format(x / 1e6)
    if abs(x) > 1e2:
        return (frmt + "K").format(x / 1e3)
    return str(x)


def _pretty_statistics(
    statistics: Dict[str, Dict[str, int]], sig_figs: int = 3, hide_zero: bool = False
) -> Dict[str, Dict[str, str]]:
    """
    Converts numeric statistics to strings with kilo/mega/giga/etc.
    labels.

    Args:
        statistics (dict(str, dict(str, int))) : the statistics to
            format. Organized as a dictionary over modules, which are
            each a dictionary over statistic types.
        sig_figs (int) : the number of significant figures for each stat
        hide_zero (bool) : if True, statistics that are zero will be
            written as an empty string. Defaults to False.

    Return:
        dict(str, dict(str, str)) : the input statistics as pretty strings
    """
    out_stats = {}
    for mod, stats in statistics.items():
        out_stats[mod] = {
            s: _format_size(val, sig_figs, hide_zero) for s, val in stats.items()
        }
    return out_stats


def _group_by_module(
    statistics: Dict[str, Dict[str, Any]]
) -> Dict[str, Dict[str, Any]]:
    """
    Converts statistics organized first by statistic type and then by module
    to statistics organized first by module and then by statistic type.

    Args:
        statistics (dict(str, dict(str, any))) : the statistics to convert

    Returns:
        dict(str, dict(str, any)) : the reorganized statistics
    """
    out_stats = defaultdict(dict)
    for stat_name, stat in statistics.items():
        for mod, val in stat.items():
            out_stats[mod][stat_name] = val
    return dict(out_stats)


def _indicate_uncalled_modules(
    statistics: Dict[str, Dict[str, str]],
    stat_name: str,
    uncalled_modules: Set[str],
    uncalled_indicator: str = "N/A",
) -> Dict[str, Dict[str, str]]:
    """
    If a module is in the set of uncalled modules, replace its statistics
    with the specified indicator, instead of using the existing string.
    Assumes the statistic is already formatting in string form.

    Args:
        statistics (dict(str, dict(str, str))) : the statistics to
            format. Organized as a dictionary over modules, which are
            each a dictionary over statistic types. Expects statistics
            have already been converted to strings.
        stat_name (str) : the name of the statistic being modified
        uncalled_modules set(str) : a set of names of uncalled modules.
        indicator (str) : the string that will be used to indicate
            unused modules. Defaults to 'N/A'.

    Returns:
        dict(str, dict(str, str)) : the modified statistics
    """

    stats_out = {mod: stats.copy() for mod, stats in statistics.items()}
    for mod in uncalled_modules:
        if mod not in stats_out:
            stats_out[mod] = {}
        stats_out[mod][stat_name] = uncalled_indicator
    return stats_out


def _remove_zero_statistics(
    statistics: Dict[str, Dict[str, int]],
    force_keep: Optional[Set[str]] = None,
    require_trivial_children: bool = False,
) -> Dict[str, Dict[str, int]]:
    """
    Any module that has zero for all available statistics is removed from the
    set of statistics. This can help declutter the reporting of statistics
    if many submodules have zero statistics. Assumes the statistics have
    a model hierarchy starting with a root that has name ''.

    Args:
        statistics (dict(str, dict(str, int))) : the statistics to
            remove zeros from. Organized as a dictionary over modules,
            which are each a dictionary over statistic types.
        force_keep (set(str) or None) : a set of modules to always keep, even
            if they are all zero.
        require_trivial_children (bool) : If True, a statistic will only
            be deleted if all its children are also deleted. Defaults to
            False.

    Returns:
        dict(str, dict(str, int)) : the input statistics dictionary,
            with submodules removed if they have zero for all statistics.
    """
    out_stats: Dict[str, Dict[str, int]] = {}
    _force_keep: Set[str] = force_keep if force_keep else set()

    def keep_stat(name: str) -> None:
        prefix = name + ("." if name else "")
        trivial_children = True
        for mod in statistics:
            # 'if mod' excludes root = '', which is never a child
            if mod and mod.count(".") == prefix.count(".") and mod.startswith(prefix):
                keep_stat(mod)
                trivial_children &= mod not in out_stats

        if (
            (not all(val == 0 for val in statistics[name].values()))
            or (name in _force_keep)
            or (require_trivial_children and not trivial_children)
        ):
            out_stats[name] = statistics[name].copy()

    keep_stat("")
    return out_stats


def _fill_missing_statistics(
    model: nn.Module, statistics: Dict[str, Dict[str, int]]
) -> Dict[str, Dict[str, int]]:
    """
    If, for a given submodule name in the model, a statistic is missing
    from statistics, fills it in with zero. This visually uniformizes
    the reporting of statistics.

    Args:
        model (nn.Module) : the model whose submodule names will be
            used to fill statistics
        statistics (dict(str, dict(str, int))) : the statistics to
            fill in missing values for. Organized as a dictionary
            over statistics, which are each a dictionary over submodules'
            names. The statistics are assumed to be formatted already
            to the desired string format for printing.

    Returns:
        dict(str, dict(str, int)) : the input statistics with missing
            values filled with zero.
    """
    out_stats = {name: stat.copy() for name, stat in statistics.items()}
    for mod_name, _ in model.named_modules():  # pyre-ignore[16]
        for stat in out_stats.values():
            if mod_name not in stat:
                stat[mod_name] = 0
    return out_stats


### Model String Printing ###


def _model_stats_str(model: nn.Module, statistics: Dict[str, Dict[str, str]]) -> str:
    """
    This produces a representation of the model much like 'str(model)'
    would, except the provided statistics are written out as additional
    information for each submodule.

    Args:
        model (nn.Module) : the model to form a representation of.
        statistics (dict(str, dict(str, str))) : the statistics to
            include in the model representations. Organized as a dictionary
            over module names, which are each a dictionary over statistics.
            The statistics are assumed to be formatted already to the
            desired string format for printing.

    Returns:
        str : the string representation of the model with the statistics
            inserted.
    """

    # Copied from nn.Module._addindent
    def _addindent(s_: str, numSpaces: int) -> str:
        s = s_.split("\n")
        # don't do anything for single-line stuff
        if len(s) == 1:
            return s_
        first = s.pop(0)
        s = [(numSpaces * " ") + line for line in s]
        s = "\n".join(s)
        s = first + "\n" + s
        return s

    def print_statistics(name: str) -> str:
        if name not in statistics:
            return ""
        printed_stats = ["{}: {}".format(k, v) for k, v in statistics[name].items()]
        return ", ".join(printed_stats)

    # This comes directly from nn.Module.__repr__ with small changes
    # to include the statistics.
    def repr_with_statistics(module: nn.Module, name: str) -> str:
        # We treat the extra repr like the sub-module, one item per line
        extra_lines = []
        extra_repr = module.extra_repr()  # pyre-ignore[16]
        printed_stats = print_statistics(name)
        # empty string will be split into list ['']
        if extra_repr:
            extra_lines.extend(extra_repr.split("\n"))
        if printed_stats:
            extra_lines.extend(printed_stats.split("\n"))
        child_lines = []
        for key, submod in module._modules.items():  # pyre-ignore[16]
            submod_name = name + ("." if name else "") + key
            submod_str = repr_with_statistics(submod, submod_name)
            submod_str = _addindent(submod_str, 2)
            child_lines.append("(" + key + "): " + submod_str)
        lines = extra_lines + child_lines

        main_str = module._get_name() + "("  # pyre-ignore[16]
        if lines:
            # simple one-liner info, which most builtin Modules will use
            if len(extra_lines) == 1 and not child_lines:
                main_str += extra_lines[0]
            else:
                main_str += "\n  " + "\n  ".join(lines) + "\n"

        main_str += ")"
        return main_str

    return repr_with_statistics(model, "")


def _get_input_sizes(iterable: Iterable[Any]) -> List[Any]:  # pyre-ignore[2,3]
    """
    Gets the sizes of all torch tensors in an iterable. If an element
    of the iterable is a non-torch tensor iterable, it recurses into
    that iterable to continue calculating sizes. Any non-iterable is
    given a size of None. The output consists of nested lists with the
    same nesting structure as the input iterables.
    """
    out_list = []
    for i in iterable:
        if isinstance(i, torch.Tensor):
            out_list.append(list(i.size()))
        elif isinstance(i, Iterable):
            sublist_sizes = _get_input_sizes(i)
            if all(j is None for j in sublist_sizes):
                out_list.append(None)
            else:
                out_list.append(sublist_sizes)
        else:
            out_list.append(None)
    return out_list


def flop_count_str(
    model: nn.Module, inputs: Tuple[Any], activations: bool = False  # pyre-ignore[2]
) -> str:
    """
    Calculates the parameters and flops of the model with the given inputs
    and returns a string representation of the model that includes the
    parameters and flops of every submodule. The string is structured
    to be similar that given by str(model), though it is not guaranteed to
    be identical in form if the default string representation of a module has
    been overridden. If a module has zero parameters and flops, statistics
    will not be reported for succinctness.

    The trace can only register the scope of a module if it is called
    directly, which means flops (and activations) arising from explicit
    calls to .forward() or to other python functions of the module will
    not be attributed to that module. Modules that are never called will
    have 'N/A' listed for their flops; this means they are either unused
    or their statistics are missing for this reason. Any such flops are still
    counted towards the parent

    Example:

    >>> import torch
    >>> import torch.nn as nn

    >>> class InnerNet(nn.Module):
    ...     def __init__(self):
    ...         super().__init__()
    ...         self.fc1 = nn.Linear(10,10)
    ...         self.fc2 = nn.Linear(10,10)
    ...     def forward(self, x):
    ...         return self.fc1(self.fc2(x))

    >>> class TestNet(nn.Module):
    ...     def __init__(self):
    ...         super().__init__()
    ...         self.fc1 = nn.Linear(10,10)
    ...         self.fc2 = nn.Linear(10,10)
    ...         self.inner = InnerNet()
    ...     def forward(self, x):
    ...         return self.fc1(self.fc2(self.inner(x)))

    >>> inputs = torch.randn((1,10))
    >>> print(model_flops_str(model, inputs))
    TestNet(
      n_params: 0.44K, n_flops: 0.4K
      (fc1): Linear(
        in_features=10, out_features=10, bias=True
        n_params: 0.11K, n_flops: 100
      )
      (fc2): Linear(
        in_features=10, out_features=10, bias=True
        n_params: 0.11K, n_flops: 100
      )
      (inner): InnerNet(
        n_params: 0.22K, n_flops: 0.2K
        (fc1): Linear(
          in_features=10, out_features=10, bias=True
          n_params: 0.11K, n_flops: 100
        )
        (fc2): Linear(
          in_features=10, out_features=10, bias=True
          n_params: 0.11K, n_flops: 100
        )
      )
    )


    Args:
        model (nn.Module) : the torch model that will be analyzed and
            returned in a string representation.
        inputs (tuple): The inputs to the model used to calculate flops.
        activations (bool) : If True, the activations of each layer will
            also be calculated and included in the representation.

    Returns:
        str : a string representation of the model with the number of
            parameters and flops included.
    """
    # cast to dict since pyre doesn't like the implicit defaultdict->dict
    params = dict(parameter_count(model))

    flop_count = FlopCountAnalysis(model=model, inputs=inputs)
    flop_count.skipped_ops_warnings(False)
    flop_count.uncalled_modules_warnings(False)
    flop_count.tracer_warnings("none")
    flops = dict(flop_count.by_module())
    stats = {"n_params": params, "n_flops": flops}

    if activations:
        act_count = ActivationCountAnalysis(model=model, inputs=inputs)
        act_count.skipped_ops_warnings(False)
        act_count.uncalled_modules_warnings(False)
        act_count.tracer_warnings("none")
        acts = dict(act_count.by_module())
        stats["n_acts"] = acts

    all_uncalled = flop_count.uncalled_modules() | (
        act_count.uncalled_modules() if activations else set()
    )
    stats = _fill_missing_statistics(model, stats)
    stats = _group_by_module(stats)
    stats = _remove_zero_statistics(stats, force_keep=all_uncalled)
    stats = _pretty_statistics(stats)
    stats = _indicate_uncalled_modules(stats, "n_flops", flop_count.uncalled_modules())
    if activations:
        stats = _indicate_uncalled_modules(
            stats, "n_acts", act_count.uncalled_modules()
        )

    input_sizes = _get_input_sizes(inputs)
    model_string = "Input sizes (torch.Tensor only): {}\n".format(input_sizes)
    if all_uncalled:
        model_string += (
            "N/A indicates a possibly missing statistic due to how "
            "the module was called. Missing values are still included "
            "in the parent's total.\n"
        )
    model_string += _model_stats_str(model, stats)
    return model_string


### Table Printing ###


def _get_single_child(
    name: str, statistics: Dict[str, Dict[str, str]]
) -> Optional[str]:
    """
    If the given module has only a single child in statistics, return it.
    Otherwise, return None.
    """
    prefix = name + ("." if name else "")
    child = None
    for mod in statistics:
        # 'if mod' excludes root = '', which is never a child
        if mod and mod.count(".") == prefix.count(".") and mod.startswith(prefix):
            if child is None:
                child = mod
            else:
                return None  # We found a second child, so return None
    return child


def _fastforward(name: str, statistics: Dict[str, Dict[str, str]]) -> str:
    """
    If the given module has only a single child and matches statistics
    with that child, get the child and the combined statistics. Then
    repeat until the condition isn't met.
    """
    single_child = _get_single_child(name, statistics)
    if single_child is None:
        return name
    if statistics[name] != statistics[single_child]:
        return name
    return _fastforward(single_child, statistics)


def _model_stats_table(
    statistics: Dict[str, Dict[str, str]],
    max_depth: int = 3,
    stat_columns: Optional[List[str]] = None,
    missing_indicator: Optional[str] = None,
) -> str:
    """
    Formats the statistics obtained from a model in a nice table.

    Args:
        statistics (dict(str, dict(str, str))) : The statistics to print.
            Organized as a dictionary over modules, then as a dictionary
            over statistics in the model. The statistics are assumed to
            already be formatted for printing.
        max_depth (int) : The maximum submodule depth to recurse to.
        stat_columns (list(str)) : Specify the order of the columns to print.
            If None, columns are found automatically from the provided
            statistics.
        missing_indicator (str or None) : If set, statistics with the
            specified string will be considered missing, and will count
            as matching a child or parent's statistics for the sake of
            skipping wrappers. When a wrapper is skipped the printed
            statistics will consist of all known non-missing statistics.

    Return:
        str : The formatted table.
    """
    if stat_columns is None:
        stat_columns = set()
        for stats in statistics.values():
            stat_columns.update(stats.keys())
        stat_columns = list(stat_columns)

    headers = ["module"] + stat_columns
    table: List[List[str]] = []

    def build_row(name: str, stats: Dict[str, str], indent_lvl: int) -> List[str]:
        indent = " " * indent_lvl
        row = [indent + name]
        for stat_name in stat_columns:  # pyre-ignore[16] Is not None at this point
            row_str = (indent + stats[stat_name]) if stat_name in stats else ""
            row.append(row_str)
        return row

    def fill(indent_lvl: int, prefix: str) -> None:
        if indent_lvl > max_depth:
            return
        for mod in statistics:
            # 'if mod' excludes root = '', which is never a child
            if mod and mod.count(".") == prefix.count(".") and mod.startswith(prefix):
                mod = _fastforward(mod, statistics)
                row = build_row(mod, statistics[mod], indent_lvl)
                table.append(row)
                fill(indent_lvl + 1, mod + ".")

    root_name = _fastforward("", statistics)
    table_name = "model" + (("." + root_name) if root_name else "")
    row = build_row(table_name, statistics[root_name], indent_lvl=0)
    table.append(row)
    root_prefix = root_name + ("." if root_name else "")
    fill(indent_lvl=1, prefix=root_prefix)

    old_ws = tabulate.PRESERVE_WHITESPACE
    tabulate.PRESERVE_WHITESPACE = True
    tab = tabulate.tabulate(table, headers=headers, tablefmt="pipe")
    tabulate.PRESERVE_WHITESPACE = old_ws
    return tab


def flop_count_table(
    model: nn.Module,
    inputs: Tuple[Any],  # pyre-ignore[2]
    max_depth: int = 3,
    activations: bool = False,
    show_param_shapes: bool = True,
) -> str:
    """
    Format the per-module parameters and flops of a model in a table.
    It looks like this:

    ::

        | model                            | #parameters or shape   | #flops    |
        |:---------------------------------|:-----------------------|:----------|
        | model                            | 34.6M                  | 65.7G     |
        |  s1                              |  15.4K                 |  4.32G    |
        |   s1.pathway0_stem               |   9.54K                |   1.23G   |
        |    s1.pathway0_stem.conv         |    9.41K               |    1.23G  |
        |    s1.pathway0_stem.bn           |    0.128K              |           |
        |   s1.pathway1_stem               |   5.9K                 |   3.08G   |
        |    s1.pathway1_stem.conv         |    5.88K               |    3.08G  |
        |    s1.pathway1_stem.bn           |    16                  |           |
        |  s1_fuse                         |  0.928K                |  29.4M    |
        |   s1_fuse.conv_f2s               |   0.896K               |   29.4M   |
        |    s1_fuse.conv_f2s.weight       |    (16, 8, 7, 1, 1)    |           |
        |   s1_fuse.bn                     |   32                   |           |
        |    s1_fuse.bn.weight             |    (16,)               |           |
        |    s1_fuse.bn.bias               |    (16,)               |           |
        |  s2                              |  0.226M                |  7.73G    |
        |   s2.pathway0_res0               |   80.1K                |   2.58G   |
        |    s2.pathway0_res0.branch1      |    20.5K               |    0.671G |
        |    s2.pathway0_res0.branch1_bn   |    0.512K              |           |
        |    s2.pathway0_res0.branch2      |    59.1K               |    1.91G  |
        |   s2.pathway0_res1.branch2       |   70.4K                |   2.28G   |
        |    s2.pathway0_res1.branch2.a    |    16.4K               |    0.537G |
        |    s2.pathway0_res1.branch2.a_bn |    0.128K              |           |
        |    s2.pathway0_res1.branch2.b    |    36.9K               |    1.21G  |
        |    s2.pathway0_res1.branch2.b_bn |    0.128K              |           |
        |    s2.pathway0_res1.branch2.c    |    16.4K               |    0.537G |
        |    s2.pathway0_res1.branch2.c_bn |    0.512K              |           |
        |   s2.pathway0_res2.branch2       |   70.4K                |   2.28G   |
        |    s2.pathway0_res2.branch2.a    |    16.4K               |    0.537G |
        |    s2.pathway0_res2.branch2.a_bn |    0.128K              |           |
        |    s2.pathway0_res2.branch2.b    |    36.9K               |    1.21G  |
        |    s2.pathway0_res2.branch2.b_bn |    0.128K              |           |
        |    s2.pathway0_res2.branch2.c    |    16.4K               |    0.537G |
        |    s2.pathway0_res2.branch2.c_bn |    0.512K              |           |
        |    ............................. |    ......              |    ...... |

    Args:
        model (nn.Module) : The model to produce statistics for
        inputs (tuple) : Sample inputs to the model used to compute flops
        max_depth (int) : The max depth of submodules to include in the
            table. Defaults to 3.
        activations (bool) : If true, include a count of activations as
            an additional column in the table. Defaults to False.
        show_param_shapes (bool) : If true, shapes for parameters will be
            included in the table. Defaults to True.

    Returns:
        str : The formatted table.

    """

    # cast to dict since pyre doesn't like the implicit defaultdict->dict
    params = dict(parameter_count(model))
    params_name = "#parameters" + (" or shape" if show_param_shapes else "")

    flop_count = FlopCountAnalysis(model=model, inputs=inputs)
    flop_count.skipped_ops_warnings(False)
    flop_count.uncalled_modules_warnings(False)
    flop_count.tracer_warnings("none")
    flops = dict(flop_count.by_module())
    flops_name = "#flops"

    stats = {params_name: params, flops_name: flops}
    stat_columns = [params_name, flops_name]

    if activations:
        act_count = ActivationCountAnalysis(model=model, inputs=inputs)
        act_count.skipped_ops_warnings(False)
        act_count.uncalled_modules_warnings(False)
        act_count.tracer_warnings("none")
        acts = dict(act_count.by_module())
        acts_name = "#activations"
        stats[acts_name] = acts
        stat_columns += [acts_name]

    all_uncalled = flop_count.uncalled_modules() | (
        act_count.uncalled_modules() if activations else set()
    )
    stats = _group_by_module(stats)
    stats = _remove_zero_statistics(
        stats, force_keep=all_uncalled, require_trivial_children=True
    )
    stats = _pretty_statistics(stats, hide_zero=True)
    stats = _indicate_uncalled_modules(stats, flops_name, flop_count.uncalled_modules())
    if activations:
        stats = _indicate_uncalled_modules(
            stats, acts_name, act_count.uncalled_modules()
        )

    # Swap in shapes for parameters or delete shapes from dict
    param_shapes: Dict[str, Tuple[int, ...]] = {
        k: tuple(v.shape) for k, v in model.named_parameters()
    }
    to_delete = []
    for mod in stats:
        if mod in param_shapes:
            if show_param_shapes:
                stats[mod][params_name] = str(param_shapes[mod])
            else:
                to_delete.append(mod)
    for mod in to_delete:
        del stats[mod]

    return _model_stats_table(
        statistics=stats,
        max_depth=max_depth,
        stat_columns=stat_columns,
        missing_indicator="N/A",
    )
