"""
Script to estimate the number of antibodies
bound to a virus at varying antibody concentrations
using a Bayesian estimator that employs MCMC
to sample from the posterior distributions.
"""

import os
from argparse import ArgumentParser

from counting_virus_binding_sites import utils
from counting_virus_binding_sites.estimator import Estimator


def main():
    # read in config
    args = parse_args()
    cfg = utils.read_config(args.config)
    result_dir = cfg["general"]["result_directory"]

    # set up the estimator
    estimator = Estimator.from_file(cfg["general"]["in"])
    estimator.create_model(cfg["priors"]["nsat"], cfg["priors"]["pi"])

    # sample from the posteriors
    kwarg_keys = ["iter", "burn", "thin", "progress_bar"]
    estimator.sample(
        os.path.join(result_dir, cfg["general"]["database"]),
        **utils.sub_dict(cfg["mcmc"], kwarg_keys)
    )

    # plot diagnostics
    if cfg["mcmc"]["diagnostics"]["plot"]:
        prefix = cfg["mcmc"]["diagnostics"]["prefix"]
        suffix = cfg["mcmc"]["diagnostics"]["suffix"]
        estimator.diagnostics(
            result_dir,
            format=cfg["mcmc"]["diagnostics"]["format"],
            prefix=prefix if prefix is not None else "",
            suffix=suffix if suffix is not None else ""
        )

    # plot posterior distributions
    if cfg["posteriors"]["plot"]:
        kwarg_keys = [
            "show_mean", "show_hdp",
            "show_median", "show_mode",
            "format"
        ]
        prefix = cfg["posteriors"]["prefix"]
        suffix = cfg["posteriors"]["suffix"]
        estimator.plot_posteriors(
            result_dir,
            **utils.sub_dict(cfg["posteriors"], kwarg_keys),
            xlim_nsat=cfg["posteriors"]["xlim"]["nsat"],
            xlim_p=cfg["posteriors"]["xlim"]["pi"],
            prefix=prefix if prefix is not None else "",
            suffix=suffix if suffix is not None else ""
        )

    # write simple statistics on the estimated parameters to file
    estimator.stats_to_file(
        os.path.join(result_dir, cfg["general"]["stats"])
    )

    # write antibody counts to file
    estimator.to_file(
        os.path.join(result_dir, cfg["general"]["out"]),
        centrality_measure=cfg["general"]["centrality_measure"]
    )


def parse_args():
    parser = ArgumentParser(
        description=(
            "Bayesian model to estimate the "
            "number of virus binding sites "
            "occupied by antibodies from "
            "fluorescence microscopy experiments "
            "at varying antibody concentrations"
        )
    )

    parser.add_argument(
        "config",
        help=(
            "configuration file in YAML format "
            "that specifies all relevant arguments - "
            "there are no hidden defaults in the code"
        )
    )

    return parser.parse_args()


if __name__ == '__main__':
    main()
