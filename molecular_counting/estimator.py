"""
Definition of a Bayesian estimator using MCMC
to sample from the posterior distributions
of estimated parameters n_sat and p_i's.

For all estimated parameters, there are
two options of choosing a prior distribution,
an informative and an uninformative one.
"""

import sys
import os
import numpy as np
import pandas as pd
import scipy
from functools import partial

import matplotlib.pyplot as plt
import seaborn as sns

import pymc

from molecular_counting import model as m
from molecular_counting.experiment import Experiment
from molecular_counting import utils


def compute_n_sat_prior(
    informative=False, poisson_mu=None,
    uniform_lower=None, uniform_upper=None
):
    """
    Compute n_sat prior.

    Note:
    There are two options for modelling n_sat:
    - uninformative: discrete uniform distribution
    - informative: Poisson distribution

    Parameters
    ----------
    informative : bool, optional (default: False)
        If True, n_sat is modelled by a
        Poisson distribution. Else, n_sat
        is modelled by a discrete uniform
        distribution.
    poisson_mu : int, optional (default: None)
        Parameter mu (i.e. mean) of
        the Poisson distribution used to
        model n_sat. Must be specified if
        `informative` is True.
    uniform_lower : int, optional (default: None)
        Lower bound of the discrete uniform
        distribution used to model n_sat.
        Must be specified if `informative`
        is False.
    uniform_upper : int, optional (default: None)
        Upper bound of the discrete uniform
        distribution used to model n_sat.
        Must be specified if `informative`
        is False.

    Returns
    -------
    pymc distribution
        Prior distribution for n_sat.
    """
    if informative:
        if poisson_mu is None:
            error_msg = (
                "If you want to use a Poisson prior for n_sat, "
                "please specify the parameter `poisson_mu`."
            )
            sys.exit(error_msg)

        return pymc.Poisson(
            "n_sat",
            mu=poisson_mu
        )

    if (
        uniform_lower is None or
        uniform_upper is None
    ):
        error_msg = (
            "If you want to use an uniform prior for n_sat, "
            "please specify the parameters `uniform_lower` "
            "and `uniform_upper`."
        )
        sys.exit(error_msg)

    return pymc.DiscreteUniform(
        "n_sat",
        lower=uniform_lower,
        upper=uniform_upper
    )


def compute_p_priors(
    experiments, informative=False,
    hill_n=None, hill_K_alpha=None,
    hill_K_beta=None
):
    """
    Compute binding probability
    priors.

    Note:
    There are two options for modelling
    antibody binding probabilities:
    - uninformative: Beta(1, 1) distributions
    - informative: priors are inferred from
    the Hill equation p = 1 / (1 + (K / c)**n)
    with unknown parameters n and K where
    n is fixed and K is modelled using a
    Gamma(alpha, beta) distribution

    Parameters
    ----------
    experiments : list of Experiment
        List of all conducted experiments
        for a particular virus-antibody pair
        at varying experimental conditions.
    informative : bool, optional (default: False)
        If True, all p's are modelled priors
        inferred from the Hill equation. Else,
        p's are modelled by Beta(1, 1) distributions.
    hill_n : float, optional (default: None)
        Parameter n of the Hill equation.
    hill_K_alpha : float, optional (default: None)
        Alpha parameter of the gamma
        distribution used to model the unknown
        parameter K of the Hill equation.
    hill_K_beta : float, optional (default: None)
        Beta parameter of the gamma
        distribution used to model the unknown
        parameter K of the Hill equation.

    Returns
    -------
    dict (key=str, value=pymc distribution)
        Prior distributions of all p's
        accessible through the experiment's id.
    """
    if not informative:
        return {
            e.experiment_id: pymc.Beta(
                e.experiment_id, alpha=1, beta=1
            ) for e in experiments
        }

    if any(
        param is None for param in [
            hill_n, hill_K_alpha, hill_K_beta
        ]
    ):
        error_msg = (
            "If you want to use informative priors for pi, "
            "please specify the parameters `hill_n`, `hill_K_alpha`, "
            "and `hill_K_beta`."
        )
        sys.exit(error_msg)

    def _p_prior_factory(
        name, concentration,
        n=hill_n,
        alpha=hill_K_alpha,
        beta=hill_K_beta,
        value=None
    ):
        """Generate a p prior for a specific concentration."""
        if value is None:
            value = np.random.rand()

        @pymc.stochastic
        def _p(value=value, c=concentration,
               n=n, alpha=alpha, beta=beta):
            x = np.exp(
                ((1 / n) * np.log((1 / value) - 1)) +
                np.log(c)
            )

            return scipy.stats.gamma.pdf(
                x, alpha, scale=1 / beta
            ) * np.abs(
                - (x / (n * value * (1 - value)))
            )

        _p.__name__ = name
        return _p

    return {
        e.experiment_id: _p_prior_factory(
            name=e.experiment_id,
            concentration=e.ab_concentration
        ) for e in experiments
    }


def compute_likelihoods(experiments, n_sat_prior, p_priors):
    """
    Compute log-likelihoods for all estimated p's.

    Parameters
    ----------
    experiments : list of Experiment
        List of all conducted experiments
        for a particular virus-antibody pair
        at varying experimental conditions.
    n_sat_prior : pymc distribution
        Prior distribution for n_sat.
    p_priors : dict (key=str, value=pymc distribution)
        Prior distributions of all p's
        accessible through the experiment's id.

    Returns
    -------
    list of pymc observed data functions
        Log-likelihoods for all estimated p's.
    """
    def _log_likelihood_factory(states, p, n_sat, fl):
        """Generate a specific log-likelihood function."""

        @pymc.observed
        def log_likelihood(
            value=states, p=p,
            n_sat=n_sat, fl=fl
        ):
            return m.log_likelihood(
                p, value, n_sat, fl
            )

        return log_likelihood

    # log-likelihoods for all binding probabilities
    return [
        _log_likelihood_factory(
            states=e.states,
            p=p_priors[e.experiment_id],
            n_sat=n_sat_prior,
            fl=e.f_labelled
        ) for e in experiments
    ]


class Estimator(object):
    """
    Bayesian estimator to estimate the
    total number of viral binding sites
    n_sat as well as antibody binding
    probabilities p_1, ..., p_m at
    antibody concentrations c_1, ..., c_m.

    Note:
    The estimator uses MCMC to sample
    from the posterior distributions.

    Attributes
    ----------
    experiments: list of Experiment
        List of all conducted experiments
        for a particular virus-antibody pair
        at varying experimental conditions.
    model : pymc.Model
        The PyMC model specifying the likelihood
        and prior distributions.
    parameters : list of str
        Key list of unknown parameters that
        are to be estimated.
    mcmc : pymc.MCMC
        The PyMC MCMC object that is used
        to sample from the posteriors.
    """
    def __init__(self, experiments):
        self.experiments = experiments
        self.model = None
        self.parameters = None
        self.mcmc = None
        self._point_estimates = None
        self._n_abs = None

    @classmethod
    def from_file(cls, filename):
        """
        Initialize an estimator from a
        file containing the experimental data.

        Parameters
        ----------
        filename : str
            Path to the file to be read in.
            Must be in comma-separated csv format.
            The following columns are required:
            data_set, ab_concentration, f_labelled,
            n_virus, and n_virus_pos.

        Returns
        -------
        Estimator
            An estimator instance.
        """
        df = pd.read_csv(filename)

        experiments = [
            Experiment(
                data_set=e.data_set,
                ab_concentration=e.ab_concentration,
                f_labelled=e.f_labelled,
                n_virus=e.n_virus,
                n_virus_pos=e.n_virus_pos
            ) for e in df.itertuples()
        ]

        return cls(experiments)

    @property
    def point_estimates(self):
        """Simple statistics on the estimated parameters."""
        if self._point_estimates is not None:
            return self._point_estimates

        if self.mcmc is not None:
            self._point_estimates = {}
            for param in self.parameters:
                trace = self.mcmc.trace(param)[:]
                stats = self.mcmc.stats(chain=-1)[param]
                self._point_estimates[param] = {
                    "mean": stats["mean"],
                    "median": np.median(trace),
                    "mode": utils.mode_from_binned_data(trace),
                    "hdp": tuple(stats["95% HPD interval"])
                }
            return self._point_estimates

        return None

    @property
    def n_abs(self):
        """Antibody counts for all conducted experiments."""
        if self._n_abs is not None:
            return self._n_abs

        if self.point_estimates is not None:
            self._n_abs = {}
            for experiment in self.experiments:
                pi = experiment.experiment_id
                self._n_abs[pi] = {
                    cm: m.n_bound_abs(
                        self.point_estimates["n_sat"][cm],
                        self.point_estimates[pi][cm]
                    ) for cm in ["mode", "mean", "median"]
                }
            return self._n_abs

        return None

    def create_model(self, n_sat_kwargs, p_kwargs):
        """
        Create Bayesian model including priors
        and likelihoods.

        Implementation note:
        This method sets the attributes
        `parameters` and `model`.

        Parameters
        ----------
        n_sat_kwargs : dict (key=str, value=any)
            Keyword arguments necessary to set
            the prior distribution for n_sat.
            This dictionary corresponds to the
            subsection `nsat` of section `priors`
            in the config.
        p_kwargs : dict (key=str, value=any)
            Keyword arguments necessary to set
            the prior distribution for all p_i's.
            This dictionary corresponds to the
            subsection `pi` of section `priors`
            in the config.
        """
        # unknown parameters of interest
        self.parameters = [e.experiment_id for e in self.experiments]
        self.parameters.append("n_sat")

        # compute priors
        n_sat_prior = compute_n_sat_prior(**n_sat_kwargs)
        p_priors = compute_p_priors(self.experiments, **p_kwargs)

        # compute likelihoods
        log_likelihoods = compute_likelihoods(
            self.experiments, n_sat_prior, p_priors
        )

        # collect all generated stochastics
        distributions = list(p_priors.values())
        distributions.append(n_sat_prior)
        distributions.extend(log_likelihoods)

        # create the model
        self.model = pymc.Model(distributions)

    def sample(
        self, dbname,
        n_runs=1, iter=10000, burn=1000, thin=10,
        gelman_rubin=False, progress_bar=False,
        **kwargs
    ):
        """
        Sample from the posteriors using MCMC.

        Implementation note:
        This method sets the attribute `mcmc`.

        Parameters
        ----------
        dbname : str
            Path to the file the pickled MCMC
            object is written to. If the path exists,
            the existing database is updated.
        n_runs : int, optional (default: 1)
            The number of times MCMC is run.
            Must be >1, if Gelman-Rubin statistic
            is used.
        iter : int, optional (default: 10000)
            The number of iterations per MCMC run.
        burn : int, optional (default: 1000)
            The number of samples discarded from
            the beginning of a parameter's trace.
        thin : int, optional (default: 10)
            Each `thin` sample is discarded to
            reduce auto-correlation.
        gelman_rubin : bool, optional (default: False)
            If True, compute the Gelman-Rubin statistic
            for each sampled parameter and print to stdout.
            n_runs must be >1.
        progress_bar : bool, optional (default: False)
            If True, show progress bar while MCMC
            samples.
        **kwargs
            Additional keyword arguments passed
            to PyMC's sample call.
        """
        if self.model is None:
            error_msg = (
                "Model doesn't exist in sampling stage. "
                "Please create the model before sampling."
            )
            sys.exit(error_msg)

        db = "pickle"
        if os.path.isfile(dbname):
            db = pymc.database.pickle.load(dbname)

        # init MCMC sampling object
        self.mcmc = pymc.MCMC(
            self.model,
            db=db,
            dbname=dbname
        )

        # sample using MCMC
        for _ in range(n_runs):
            self.mcmc.sample(
                iter=iter,
                burn=burn,
                thin=thin,
                progress_bar=progress_bar,
                **kwargs
            )

        # close the database file
        self.mcmc.db.close()

        # compute Gelman-Rubin statistic
        if gelman_rubin:
            if n_runs < 2:
                print(
                    "The Gelman-Rubin statistic requires",
                    "multiple MCMC runs.",
                    file=sys.stderr
                )
            else:
                print("Gelman-Rubin statistics:")
                for param in self.parameters:
                    gr = pymc.gelman_rubin(self.mcmc)[param]
                    print(f"\t{param} : {gr}")

    def diagnostics(
        self, directory, format="pdf",
        prefix="", suffix="_diagnostics"
    ):
        """
        Plot simple diagnostics of the sample
        process for each parameter.

        Note:
        This method generates a figure
        for each sampled parameter. The
        figure includes the parameter's
        trace, auto-correlation, and
        histogram.

        Note:
        Figures will be saved to the path
        `directory`/`prefix``parameter_id``suffix`.`format`,
        e.g. /home/sophia/test_c0.1_fl0.01_diagnostics.pdf.

        Parameters
        ----------
        directory : str
            Path to the directory the
            figures will be saved to.
        format : str, optional (default: "pdf")
            The file format.
        prefix : str, optional (default: "")
            Prefix of the filename.
        suffix : str, optional (default: "_diagnostics")
            Suffix of the filename.
        """
        pymc.Matplot.plot(
            self.mcmc,
            path=directory,
            suffix=suffix,
            format=format,
            verbose=0
        )

        # make sure all open figures are closed
        plt.close("all")

        # add prefix to filename if requested
        if prefix:
            filenames = [
                param + suffix + "." + format
                for param in self.parameters
            ]
            for filename in filenames:
                current = os.path.join(directory, filename)
                prefixed = os.path.join(directory, prefix + filename)
                os.system(f"mv {current} {prefixed}")

    def _plot_posterior(
        self, parameter,
        show_mean=False, show_hdp=True,
        show_median=False, show_mode=True,
        xlim=None, **kwargs
    ):
        """
        Generate posterior plot for
        a given parameter.

        Parameters
        ----------
        parameter : str
            The parameter's id whose posterior
            distribution is to be plotted.
        show_mean : bool, optional (default: False)
            If True, show the mean of the
            empirical distribution.
        show_hdp : bool, optional (default: True)
            If True, show the 95% HDP interval
            of the distribution.
        show_median : bool, optional (default: False)
            If True, show the median of the
            distribution.
        show_mode : bool, optional (default: True)
            If True, show the mode of the
            distribution.
        xlim : tuple of float, optional (default: None)
            Limits on x-axis. If None, no
            constraints are applied.
        **kwargs
            Additional keyword arguments passed
            to seaborn's `distplot` call.

        Returns
        -------
        matplotlib axis
            The generated axis.
        """
        # plot histogram and KDE
        ax = plt.gca()
        trace = self.mcmc.trace(parameter)[:]
        sns.distplot(trace, ax=ax, **kwargs)

        # vertical line to show measures of centrality
        vline = partial(ax.axvline, color="0.1", lw=0.8)

        # is current parameter n_sat
        is_nsat = parameter == "n_sat"

        # show mean of the distribution
        if show_mean:
            mean = self.point_estimates[parameter]["mean"]
            readable = utils.to_readable_str(mean, is_nsat)
            vline(mean, ls=":", label=f"Mean ({readable})")

        # show median of the distribution
        if show_median:
            median = self.point_estimates[parameter]["median"]
            readable = utils.to_readable_str(median, is_nsat)
            vline(median, ls="-.", label=f"Median ({readable})")

        # show mode of the distribution
        if show_mode:
            mode = self.point_estimates[parameter]["mode"]
            readable = utils.to_readable_str(mode, is_nsat)
            vline(mode, ls="--", label=f"Mode ({readable})")

        # show 95% HDP interval
        if show_hdp:
            hdp = self.point_estimates[parameter]["hdp"]
            readable = [utils.to_readable_str(x, is_nsat) for x in hdp]
            ax.axvspan(
                *hdp, facecolor="0.1", alpha=0.1,
                label="95% HPD interval ({}-{})".format(*readable)
            )

        if xlim is not None:
            ax.set_xlim(*xlim)
        ax.set_xlabel("Parameter")
        ax.set_ylabel("Density")
        ax.set_title(parameter)
        ax.legend()

        return ax

    def plot_posteriors(
        self, directory,
        show_mean=False, show_hdp=True,
        show_median=False, show_mode=True,
        xlim_nsat=None, xlim_p=None, format="pdf",
        prefix="", suffix="_posterior"
    ):
        """
        Plot the kernel density estimate and
        histogram of the posterior distribution
        for each estimated parameter.

        Note:
        Figures will be saved to the path
        `directory`/`prefix``parameter_id``suffix`.`format`,
        e.g. /home/sophia/test_c0.1_fl0.01_posterior.pdf.

        Parameters
        ----------
        directory : str
            Path to the directory the
            figures will be saved to.
        show_mean : bool, optional (default: False)
            If True, show the mean of the
            posterior distribution.
        show_hdp : bool, optional (default: True)
            If True, show the 95% HDP interval
            of the posterior distribution.
        show_median : bool, optional (default: False)
            If True, show the median of the
            posterior distribution.
        show_mode : bool, optional (default: True)
            If True, show the mode of the
            posterior distribution.
        xlim_nsat : tuple of float, optional (default: None)
            Limits on x-axis for n_sat. If None, no
            constraints are applied.
        xlim_p : tuple of float, optional (default: None)
            Limits on x-axis for p's. If None, no
            constraints are applied.
        format : str, optional (default: "pdf")
            The file format.
        prefix : str, optional (default: "")
            Prefix of the filename.
        suffix : str, optional (default: "_diagnostics")
            Suffix of the filename.
        """
        # template for output paths
        out_template = os.path.join(
            directory,
            prefix + "{}" + suffix + "." + format
        )

        # plot posterior for each parameter
        for param in self.parameters:
            if param == "n_sat":
                xlim = xlim_nsat
            else:
                xlim = xlim_p

            self._plot_posterior(
                param,
                show_mean=show_mean,
                show_hdp=show_hdp,
                show_median=show_median,
                show_mode=show_mode,
                xlim=xlim
            )

            # save to file
            out = out_template.format(param)
            plt.savefig(out, bbox_inches="tight")
            plt.clf()

    def stats_to_file(self, filename):
        """
        Write simple statistics for all
        estimated parameters to file.

        Parameters
        ----------
        filename : str
            Path to out file.
        """
        stats = self.mcmc.stats(chain=-1)
        for param in self.parameters:
            stats[param]["median"] = self.point_estimates[param]["median"]
            stats[param]["mode"] = self.point_estimates[param]["mode"]

        utils.dump_json(filename, stats)

    def to_file(self, filename, centrality_measure="mode"):
        """
        Write antibody counts to csv file.

        Parameters
        ----------
        filename : str
            Path to out file.
        centrality_measure : {"mode", "mean", "median"}, optional (default: "mode")
            The centrality measure used to
            compute antibody counts.
        """
        df = pd.DataFrame(self.point_estimates).T
        df["n_abs"] = pd.Series(
            {**{p: d[centrality_measure] for p, d in self.n_abs.items()},
             **{"n_sat": None}}
        )
        df = df[[centrality_measure, "n_abs"]].sort_index()
        df.to_csv(filename)

    def _reset(self):
        """Reset the estimator only keeping experimental data."""
        self.model = None
        self.parameters = None
        self.mcmc = None
        self._point_estimates = None
        self._n_abs = None
