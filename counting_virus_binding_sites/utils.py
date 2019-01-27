"""
Miscellaneous helper functions.
"""

import sys
import yaml
import json
import numpy as np


class NumpyEncoder(json.JSONEncoder):
    """JSON encoder capable of handling numpy objects.

    From here:
    https://stackoverflow.com/questions/26646362/numpy-array-is-not-json-serializable
    """
    def default(self, obj):
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        return json.JSONEncoder.default(self, obj)


def dump_json(filename, obj, pretty_printing=True):
    """Dump content to JSON file.

    Note:
    This function allows to dump objects
    that include numpy arrays.

    Parameters
    ----------
    filename : str
        The path to the json file
        that is written.
    obj : any
        The object that is dumped
        to a JSON file.
    pretty_printing : bool, optional (default: True)
        If True, JSON content is pretty
        printed with an indent of 4.
    """
    with open(filename, "w") as f:
        json.dump(
            obj, f,
            cls=NumpyEncoder,
            indent=4 if pretty_printing else None
        )


def read_config(filename):
    """
    Read a config file.

    Parameters
    ----------
    filename : str
        The path to the config
        file to be read in.

    Returns
    -------
    dict
        Config content.
    """
    with open(filename) as f:
        return yaml.safe_load(f)


def write_config(filename, config):
    """
    Write a config file.

    Parameters
    ----------
    filename : str
        The path to the config
        file that is written.
    config : dict
        The config content.
    """
    with open(filename, "w") as f:
        f.write(
            yaml.dump(config, default_flow_style=False)
        )


def mode_from_binned_data(trace):
    """
    Compute the mode from continuous
    empirical data.

    Note:
    After dividing the data into bins
    of equal width, this formula is applied:
    mode = L + (f_m - f_p) * w / (2*f_m - f_p - f_s)
    where L is the lower limit of the modal
    class, f_m, f_p, and f_s are frequencies of
    the modal class, the predecessor and
    successor of the model class, respectively,
    and w is the bin width.

    Note:
    If the computed bins are not of
    equal width (within some allowance),
    then simply the mean of the modal
    class is returned as mode.

    Note:
    It is assumed the empirical distribution
    of given values is uni-modal.

    Parameters
    ----------
    trace : list of float
        Series of numerical values
        that the mode is computed from.

    Returns
    -------
    float
        The mode of the given trace.
    """
    # compute frequency in bins
    hist, bin_edges = np.histogram(trace, bins="auto")

    # find bin with the highest frequency
    # (assumes the distribution is uni-modal)
    modal_class_index = list(hist).index(max(hist))

    # compute all bin widths
    bin_widths = [
        round(bin_edges[i] - bin_edges[i - 1], 4)
        for i in range(1, len(bin_edges))
    ]

    # if bin widths are not equal,
    # return the mean of the modal class as mode
    bin_width = bin_widths[0]
    if bin_widths.count(bin_width) != len(bin_widths):
        print(
            "Bins are not of equal width. "
            "Mode is approximated by the "
            "mean of the modal class.",
            file=sys.stderr
        )

        edges = bin_edges[
            modal_class_index:
            modal_class_index + 2
        ]

        return np.mean(edges)

    # lower boundary and frequency of the modal class
    modal_class_lower = bin_edges[modal_class_index]
    modal_class_freq = hist[modal_class_index]

    # frequencies of preceding and succeeding class
    modal_pred_freq, modal_succ_freq = 0, 0
    if modal_class_index > 0:
        modal_pred_freq = hist[modal_class_index - 1]
    if modal_class_index < len(hist) - 1:
        modal_succ_freq = hist[modal_class_index + 1]

    # compute the mode
    mode = modal_class_lower + (
        (modal_class_freq - modal_pred_freq) * bin_width /
        (2 * modal_class_freq - modal_pred_freq - modal_succ_freq)
    )

    return mode


def sub_dict(d, keep):
    """
    Keep only entries with the
    given keys in the dictionary.

    Parameters
    ----------
    d : dict
        The dictionary.
    keep : list of any
        The keys to keep.

    Returns
    -------
    dict
        Dictionary with the
        given keys.
    """
    return {
        key: value
        for key, value in d.items()
        if key in keep
    }


def to_readable_str(value, is_int):
    """
    Transform a given numerical value
    to a human readable string.

    Parameters
    ----------
    value : {int, float}
        A numerical value.
    is_int : bool
        Indicates if the given
        value is to be interpreted
        as an integer.

    Returns
    -------
    str
        The given numerical value
        as human readable string.
    """
    if is_int and value % 1 == 0:
        return str(int(value))
    return f"{value:.2f}"
