import logging
import multiprocessing
import sys
from typing import List, Tuple

import numpy as np
import psutil
import pyarrow as pa
from adam_core.coordinates import CartesianCoordinates, CoordinateCovariances, Origin
from adam_core.coordinates.transform import cartesian_to_frame
from adam_core.orbit_determination.evaluate import (
    FittedOrbitMembers,
    FittedOrbits,
    OrbitDeterminationObservations,
)
from adam_core.orbit_determination.orbit_fitter import OrbitFitter
from adam_core.time import Timestamp
from layup.orbitfit import orbitfit
from layup_cmdline import bootstrap

logger = logging.getLogger(__name__)


def _compute(fitter, object_id, observations, returns):
    """Wrapper around orbit fit method to allow for timeout."""
    returns["orbits"] = fitter._try_initial_fit(object_id, observations)


class LayupOrbitFitter(OrbitFitter):
    """Implementation of OrbitFitter using Layup."""

    WHOLE = [(0.0, 1.0)]
    HALVES = [(0, 0.5), (0.5, 1.0)]
    QUARTERS = [(i * 0.25, (i + 1) * 0.25) for i in range(4)]
    TENTHS = [(i * 0.1, (i + 1) * 0.1) for i in range(10)]

    def __init__(
        self,
        *args: object,  # Generic type for arbitrary positional arguments
        fractions: List[Tuple[float, float]] = WHOLE + HALVES + QUARTERS + TENTHS,
        timeout_s: float = 30,
        min_slice_size: int = 10,
        **kwargs: object,  # Generic type for arbitrary keyword arguments
    ) -> None:
        """Constructor for Layup wrapper.

        Parameters:
        -----------
        fractions: List[Tuple[float, float]], default = WHOLE + HALVES + QUARTERS + TENTHS
          List of pairs (min_fraction, max_fraction) for slicing the input set. The pairs
          are tried in order until a solution is found. Typically, the first pair is WHOLE,
          which is (0.0, 1.0), to try the complete input set.
        timeout_s: float, default 30
          Timeout is seconds for processing of each pair.
        min_slice_size: int, default 10
          The minimum size of the slice to consider for processing. Pairs that result in
          fewer input points than this limit are ignored. The limit is NOT applied to the
          first pair in the list of fractions.
        """
        super().__init__(*args, **kwargs)
        for pair in fractions:
            assert (
                pair[0] >= 0.0 and pair[1] <= 1.0 and pair[0] < pair[1]
            ), f"Fractions should be in [0,1] range with start < end. Got {pair}"
        self.fractions = fractions
        self.timeout_s = timeout_s
        self.min_slice_size = min_slice_size

    def __getstate__(self):
        state = self.__dict__.copy()
        return state

    def __setstate__(self, state):
        self.__dict__.update(state)

    @classmethod
    def bootstrap(cls):
        """Call Layup's bootstrap to download required files.

        This may take a while the first time it's called, but exits very quickly if
        the cache is already initialized.
        The bootstrap method parses command line arguments, so we bypass them all here.
        """
        original_args = sys.argv
        sys.argv = sys.argv[:1]
        bootstrap.main()
        sys.argv = original_args

    def initial_fit(
        self,
        object_id: str | pa.LargeStringScalar,
        observations: OrbitDeterminationObservations,
    ) -> Tuple[FittedOrbits, FittedOrbitMembers]:
        if observations is None or len(observations) == 0:
            logger.error(f"No observation provided for object {object_id}")
            return FittedOrbits.empty(), FittedOrbitMembers.empty()
        logger.info(
            f"Initial propagation using Layup for {object_id} using {len(observations)} observations"
        )

        if isinstance(object_id, pa.LargeStringScalar):
            object_id = object_id.as_py()

        total = len(observations)
        orbits = FittedOrbits.empty()
        solution = np.array([True] * len(observations))
        idx = 0
        returns = multiprocessing.Manager().dict()
        while len(orbits) == 0 and idx < len(self.fractions):
            logger.error(f"Trying {self.fractions[idx]} fraction")
            start_idx = int(total * self.fractions[idx][0])
            end_idx = int(total * self.fractions[idx][1])
            # Don't apply min slice to the first fraction, which is usually "whole"
            if idx > 0 and end_idx - start_idx < self.min_slice_size:
                logger.debug(
                    f"Skipping slice {self.fractions[idx]} with fewer than {self.min_slice_size} elements"
                )
                idx += 1
                continue
            # Layup doesn't have timeout parameter, so wrap it here in a process.
            proc = multiprocessing.Process(
                target=_compute,
                args=(self, object_id, observations[start_idx:end_idx], returns),
            )
            proc.start()
            proc.join(self.timeout_s)
            if proc.is_alive():
                logger.info(
                    f"Timed out for {self.fractions[idx]} fraction with timeout {self.timeout_s}"
                )
                # Layup uses C/C++ code, which ends up in child processes. To kill it we need to
                # kill the whole tree.
                try:
                    parent = psutil.Process(proc.pid)
                    children = parent.children(recursive=True)
                    for child in children:
                        child.kill()
                    parent.kill()
                except psutil.NoSuchProcess:
                    pass
                proc.join()
                proc.close()
            else:
                orbits = returns["orbits"]
                logger.info(
                    f"Found solution for {self.fractions[idx]}, indexes {start_idx}:{end_idx} out of {total}"
                )
                if len(orbits) > 0:
                    solution[:start_idx] = False
                    solution[end_idx:] = False
                returns.clear()
            idx += 1
        clean_id = object_id.replace(" ", "").replace("/", "")
        if len(orbits) == 0:
            members = FittedOrbitMembers.empty()
        else:
            members = FittedOrbitMembers.from_kwargs(
                orbit_id=np.full(len(observations), clean_id, dtype="object"),
                obs_id=observations.id,
                # not setting residuals here
                solution=solution,
                outlier=~solution,
            )
        return orbits, members

    def _try_initial_fit(
        self,
        object_id: str,
        observations: OrbitDeterminationObservations,
    ) -> FittedOrbits:
        """Helper method to actually run Layup."""
        if observations is None or len(observations) == 0:
            return FittedOrbits.empty()
        # Create input, column names are required
        input_data = np.array(
            list(
                zip(
                    [object_id] * len(observations),
                    observations.coordinates.lon.to_pylist(),
                    observations.coordinates.lat.to_pylist(),
                    observations.observers.code.to_pylist(),
                    observations.coordinates.time.rescale("tdb")
                    .to_iso8601()
                    .to_pylist(),
                )
            ),
            dtype=[
                ("id", "O"),
                ("ra", "<f8"),
                ("dec", "<f8"),
                ("stn", "O"),
                ("obsTime", "O"),
            ],
        )

        # Run the fitter. Note that Layup can do several objects at once, but we are
        # doing just one for now
        fitted_orbits_complete = orbitfit(
            input_data,
            cache_dir=None,
            primary_id_column_name="id",
        )
        if fitted_orbits_complete is None or np.isnan(fitted_orbits_complete["x"][0]):
            logger.error(
                f"No solution found for {object_id} with {len(observations)} observations"
            )
            return FittedOrbits.empty()
        assert (
            len(fitted_orbits_complete) == 1
        ), f"There should be exactly one orbit for now, got {len(fitted_orbits_complete)}"
        logger.error(
            f"Layup output { {name: fitted_orbits_complete[name][0] for name in fitted_orbits_complete.dtype.names} }"
        )

        if fitted_orbits_complete["flag"][0] != 0:
            logger.error(
                f"Got non-zero flag {fitted_orbits_complete['flag'][0]} for {object_id}"
            )
            return FittedOrbits.empty()

        # The result can be in either ecliptic or equatorial frame.
        # We want the final result to be in ecliptic.
        assert fitted_orbits_complete["FORMAT"][0] in [
            "BCART",
            "BCART_EQ",
        ], f"Only support cartesian for now, got {fitted_orbits_complete['FORMAT'][0]}"
        is_equatorial = fitted_orbits_complete["FORMAT"][0] == "BCART_EQ"

        # Layup doesn't mark outliers, so assume all input was used
        times = observations.coordinates.time.mjd()
        arc_length = np.max(times) - np.min(times)

        # The output contains the whole matrix, not just upper triangular
        covariances = np.zeros((6, 6), dtype=np.float64)
        for i in range(6):
            for j in range(6):
                covariances[i, j] = fitted_orbits_complete[f"cov_{i}_{j}"][0]

        cartesian_coordinates = CartesianCoordinates.from_kwargs(
            x=fitted_orbits_complete["x"],
            y=fitted_orbits_complete["y"],
            z=fitted_orbits_complete["z"],
            vx=fitted_orbits_complete["xdot"],
            vy=fitted_orbits_complete["ydot"],
            vz=fitted_orbits_complete["zdot"],
            time=Timestamp.from_mjd(
                fitted_orbits_complete["epochMJD_TDB"], scale="tdb"
            ),
            origin=Origin.from_kwargs(code=["SUN"]),
            frame="equatorial" if is_equatorial else "ecliptic",
            covariance=CoordinateCovariances.from_matrix(
                np.reshape(covariances, (1, 6, 6))
            ),
        )
        if is_equatorial:
            cartesian_coordinates = cartesian_to_frame(
                cartesian_coordinates, "ecliptic"
            )

        orbit = FittedOrbits.from_kwargs(
            orbit_id=[object_id],
            object_id=[object_id],
            coordinates=cartesian_coordinates,
            arc_length=[arc_length],
            num_obs=[len(observations)],
            chi2=[0],  # not nullable
            reduced_chi2=[0],  # not nullable
        )
        return orbit
