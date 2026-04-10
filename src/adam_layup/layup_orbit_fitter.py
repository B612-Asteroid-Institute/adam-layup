import logging
import sys
from typing import Tuple

import numpy as np
import pyarrow as pa
import quivr as qv
from adam_core.coordinates import CartesianCoordinates, CoordinateCovariances, Origin
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


class LayupInput(qv.Table):
    """Internal class to facilitate writing CSV table for inputs."""

    id = qv.LargeStringColumn()
    ra = qv.Float64Column()
    dec = qv.Float64Column()
    stn = qv.LargeStringColumn()
    obsTime = qv.LargeStringColumn()


class LayupOrbitFitter(OrbitFitter):
    """Implementation of OrbitFitter using Layup."""

    def __init__(
        self,
        *args: object,  # Generic type for arbitrary positional arguments
        **kwargs: object,  # Generic type for arbitrary keyword arguments
    ) -> None:
        super().__init__(*args, **kwargs)

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

        # Create input, column names are required
        obsTime = observations.coordinates.time.rescale("tdb").to_iso8601().to_pylist()
        inputs = list(
            zip(
                [object_id] * len(observations),
                observations.coordinates.lon.to_pylist(),
                observations.coordinates.lat.to_pylist(),
                observations.observers.code.to_pylist(),
                obsTime,
            )
        )
        complete_input_data = np.array(
            inputs,
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
            complete_input_data,
            cache_dir=None,
            primary_id_column_name="id",
        )
        if fitted_orbits_complete is None or np.isnan(fitted_orbits_complete["x"][0]):
            logger.error(
                f"No solution found for {object_id} with {len(observations)} observations"
            )
            return FittedOrbits.empty(), FittedOrbitMembers.empty()
        assert (
            len(fitted_orbits_complete) == 1
        ), f"There should be exactly one orbit for now, got {len(fitted_orbits_complete)}"

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
            frame="ecliptic",
            covariance=CoordinateCovariances.from_matrix(
                np.reshape(covariances, (1, 6, 6))
            ),
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
        clean_id = object_id.replace(" ", "").replace("/", "")
        members = FittedOrbitMembers.from_kwargs(
            orbit_id=np.full(len(observations), clean_id, dtype="object"),
            obs_id=observations.id,
            # not setting residuals here
            solution=[True] * len(observations),
            outlier=[False] * len(observations),
        )

        return orbit, members
