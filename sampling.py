"""
Abstract base class for statistical resampling arrays.

Defines the common interface for ``BootstrapSamples`` and ``JackknifeSamples``.
Both subclasses are NumPy ndarray subclasses that hold resampled data and
enforce the use of their own ``mean()`` and ``error()`` estimators instead of
the standard NumPy equivalents.

"""

from abc import ABCMeta, abstractmethod
import numpy as np


class AbstractSamples(np.ndarray, metaclass=ABCMeta):
    """Abstract base class for resampling arrays (bootstrap, jackknife).

    Subclasses store resampled data as a NumPy array and must implement
    the statistical estimators ``mean()`` and ``error()``.  Direct use of
    ``np.mean``, ``np.std``, and ``np.average`` is blocked to prevent
    accidental misuse — call the concrete methods instead.

    .. ldt-id:: SAMP-AbstractSamples
    """

    def __array_finalize__(self, obj):
        if obj is None:
            return

    def __array_function__(self, func, types, args, kwargs):
        """Block ``np.mean``, ``np.std``, and ``np.average``.

        Subclasses enforce their own estimators; calling the NumPy
        equivalents directly on a resampling array is almost always a
        mistake.

        .. ldt-id:: SAMP-AbstractSamples-__array_function__
        """
        if func in {np.mean, np.std, np.average}:
            raise TypeError(
                f"{func.__name__} is disabled for {type(self).__name__}. "
                "Use .mean() or .error() instead."
            )
        return super().__array_function__(func, types, args, kwargs)

    def to_numpy(self) -> np.ndarray:
        """Return a plain ``np.ndarray`` view of this object.

        .. ldt-id:: SAMP-AbstractSamples-to_numpy
        """
        return self.view(np.ndarray)

    @abstractmethod
    def mean(self) -> np.ndarray:
        """Estimate of the mean of the underlying distribution.

        The precise definition depends on the resampling scheme:

        - **Bootstrap**: biased mean over the bootstrap replicates (rows 1…N_bts).
        - **Jackknife**: unbiased leave-one-out mean.

        .. ldt-id:: SAMP-AbstractSamples-mean
        """

    @abstractmethod
    def error(self) -> np.ndarray:
        """Estimate of the standard error on the mean.

        The precise definition depends on the resampling scheme:

        - **Bootstrap**: standard deviation of the bootstrap replicates.
        - **Jackknife**: jackknife standard error (scaled by √(N−1)).

        .. ldt-id:: SAMP-AbstractSamples-error
        """
