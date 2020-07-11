"""
This module implements a set of functions
that constitute a statistical model that
allows to count antibody binding sites on a
virus from fluorescence microscopy imagery.

This script implements the statistical model
described in

TODO: add reference
"""

import numpy as np
from scipy.special import comb


def prob_k(k, n_sat, p):
    """
    Compute the probability that exactly
    k of n_sat binding sites of a virus
    are occupied by antibodies.

    Note:
    Computes P(K=k) where K ∼ Bin(n_sat, p).

    Parameters
    ----------
    k : int
        The number of occupied binding sites
        of a virus.
    n_sat : int
        The total number of binding sites
        of a virus.
    p : float
        The probability of a viral binding site
        to be occupied.

    Returns
    -------
    float
        The probability that exactly k of
        n_sat binding sites of a virus are
        occupied by antibodies.
    """
    return (
        comb(n_sat, k) *
        np.power(p, k) *
        np.power(1 - p, n_sat - k)
    )


def prob_neg_state_cond_k(k, fl):
    """
    Compute the probability that given the virus
    binds to k antibodies, exactly zero of them
    are fluorescently labelled.

    Note:
    Computes P(S=0|K=k) where S|K=k ∼ Bin(k, fl).

    Parameters
    ----------
    k : int
        The number of occupied binding sites
        of a virus.
    fl : float
        The proportion of antibodies labelled for
        fluorescence.

    Returns
    -------
    float
        The probability that given the virus binds
        to k antibodies, exactly zero of them are
        fluorescently labelled.
    """
    return np.power(1 - fl, k)


def prob_neg_state(p, n_sat, fl):
    """
    Compute the probability of a virus not to
    interact with any labelled antibody.

    Note:
    Computes P(S=0).

    Parameters
    ----------
    p : float
        The probability of a viral binding site
        to be occupied.
    n_sat : int
        The total number of binding sites
        of a virus.
    fl : float
        The proportion of antibodies labelled for
        fluorescence.

    Returns
    -------
    float
        The probability of a virus not to interact
        with any labelled antibody.
    """
    return sum(
        prob_neg_state_cond_k(k, fl) *
        prob_k(k, n_sat, p)
        for k in range(n_sat + 1)
    )


def log_likelihood(p, states, n_sat, fl):
    """
    Compute the log-likelihood of p, the
    probability of a viral binding site
    to be occupied.

    Note:
    Computes log L(p;s).

    Parameters
    ----------
    p : float
        The probability of a viral binding site
        to be occupied.
    states : list of {0,1}
        The list of virus states where a state
        can be positive (1) or negative (0).
        A virus is said to be in a positive state
        if it has been observed to bind at least
        on labelled antibody, else it is assigned
        a negative state.
    n_sat : int
        The total number of binding sites
        of a virus.
    fl : float
        The proportion of antibodies labelled
        for fluorescence.

    Returns
    -------
    float
        The log-likelihood of p, the probability
        of a viral binding site to be occupied.
    """
    # pre-compute probabilities
    prob_neg = prob_neg_state(p, n_sat, fl)
    prob_pos = 1 - prob_neg

    return sum(
        np.log(prob_neg) if state == 0
        else np.log(prob_pos)
        for state in states
    )


def n_bound_abs(n_sat, p):
    """
    Compute the expected number of
    antibodies bound to a virus.

    Note:
    Computes E[K].

    Parameters
    ----------
    n_sat : int
        The total number of binding sites
        of a virus.
    p : float
        The probability of a viral binding site
        to be occupied.

    Returns
    -------
    int
        The expected number of antibodies
        bound to a single virus.
    """
    return n_sat * p
