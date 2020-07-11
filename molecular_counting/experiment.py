"""
Definition of a fluorescence microscopy
experiment investigating virus-antibody
interactions.
"""

import numpy as np


class Experiment(object):
    """
    A fluorescence microscopy experiment
    investigating virus-antibody interactions.

    Attributes
    ----------
    data_set: str
        Unique identifier of the data set
        the described experiment belongs to.
    ab_concentration: float
        The antibody concentration at which
        the experiment was conducted.
    f_labelled : float
        The proportion of antibodies labelled for
        fluorescence.
    n_virus : int
        The total number of viruses used in the
        experiment.
    n_virus_pos : int
        The number of viruses that have been
        labelled positive in the experiment
        (n_virus_pos <= n_virus).
    states : np.array
        Array of size n_virus where n_virus_pos
        positions are filled with 1's, the rest
        are 0's.
    """
    def __init__(
        self,
        data_set=None,
        ab_concentration=None,
        f_labelled=None,
        n_virus=None,
        n_virus_pos=None,
        states=None
    ):
        self.data_set = data_set
        self.ab_concentration = ab_concentration
        self.f_labelled = f_labelled
        self._n_virus = n_virus
        self._n_virus_pos = n_virus_pos
        self._states = states
        self._experiment_id = None

    @property
    def n_virus(self):
        """Total number of viruses in an experiment."""
        if self._n_virus is not None:
            return self._n_virus

        if self._states is not None:
            self._n_virus = len(self.states)
            return self._n_virus

        return None

    @n_virus.setter
    def n_virus(self, n_virus):
        self._n_virus = n_virus

    @property
    def n_virus_pos(self):
        """Number of positive viruses in an experiment."""
        if self._n_virus_pos is not None:
            return self._n_virus_pos

        if self._states is not None:
            self._n_virus_pos = int(self.states.sum())
            return self._n_virus_pos

        return None

    @n_virus_pos.setter
    def n_virus_pos(self, n_virus_pos):
        self._n_virus_pos = n_virus_pos

    @property
    def states(self):
        """Binary array specifying virus states."""
        if self._states is not None:
            return self._states

        if (
            self.n_virus is not None and
            self.n_virus_pos is not None
        ):
            self._states = np.append(
                np.ones(self.n_virus_pos),
                np.zeros(self.n_virus - self.n_virus_pos)
            )
            return self._states

        return None

    @states.setter
    def states(self, states):
        self._states = states

    @property
    def experiment_id(self):
        if self._experiment_id is not None:
            return self._experiment_id

        if all(
            attr is not None for attr in [
                self.data_set,
                self.ab_concentration,
                self.f_labelled
            ]
        ):
            self._experiment_id = (
                f"e{self.data_set}_"
                f"c{self.ab_concentration}_"
                f"fl{self.f_labelled}"
            )
            return self._experiment_id

        return None

    def __str__(self):
        return (
            f"Experiment("
            f"experiment_id={self.experiment_id}, "
            f"data_set={self.data_set}, "
            f"ab_concentration={self.ab_concentration}, "
            f"f_labelled={self.f_labelled}, "
            f"n_virus={self.n_virus}, "
            f"n_virus_pos={self.n_virus_pos})"
        )

    def __repr__(self):
        return (
            self.__str__()[: -1] +
            f", states={self.states})"
        )
