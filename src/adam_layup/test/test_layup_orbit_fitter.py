import pickle

import numpy as np
import pyarrow as pa
import pytest
from adam_core.coordinates import SphericalCoordinates
from adam_core.coordinates.origin import Origin
from adam_core.observers import Observers
from adam_core.orbit_determination.evaluate import (
    OrbitDeterminationObservations,
    OrbitDeterminationPhotometry,
)
from adam_core.time import Timestamp

from ..layup_orbit_fitter import LayupOrbitFitter


@pytest.fixture
def real_data():
    # Actual observations for "2009 JY22"
    obstimes = Timestamp.from_kwargs(
        days=[54952, 54952, 54952, 54952, 54977, 54977, 54977, 54977],
        nanos=[
            15930432000000,
            16879968000000,
            17813088000000,
            18760032000000,
            14122080000000,
            14906592000000,
            15680736000000,
            16459200000000,
        ],
        scale="utc",
    )
    obscodes = ["G96", "G96", "G96", "G96", "G96", "G96", "G96", "G96"]
    lon = [
        173.174080,
        173.173500,
        173.173170,
        173.172420,
        174.067330,
        174.068380,
        174.069290,
        174.070500,
    ]
    lat = [
        7.762110,
        7.760780,
        7.759580,
        7.758310,
        4.412810,
        4.411390,
        4.410170,
        4.408890,
    ]
    obsids = [f"KG0CNl00000055470100001d{i}" for i in range(8)]
    bands = ["V", "V", "V", "V", "V", "V", "V", "V"]
    mags = [21.1, 20.5, 21.2, 21.1, 21.9, 21.2, 21.8, 21.4]

    coords = SphericalCoordinates.from_kwargs(
        lon=lon,
        lat=lat,
        time=obstimes,
        origin=Origin.from_kwargs(code=["SUN"] * 8),
        frame="equatorial",
    )
    observers = Observers.from_codes(codes=obscodes, times=obstimes)

    photometry = OrbitDeterminationPhotometry.from_kwargs(
        mag=mags,
        band=bands,
    )

    observations = OrbitDeterminationObservations.from_kwargs(
        id=obsids,
        coordinates=coords,
        observers=observers,
        photometry=photometry,
    )
    return observations


def test_pickle():
    fitter = LayupOrbitFitter()
    saved = pickle.dumps(fitter)
    pickle.loads(saved)


def test_success(real_data):
    LayupOrbitFitter.bootstrap()
    observations = real_data
    fitter = LayupOrbitFitter()
    object_id = "2009 JY22"
    # Make sure it works with PA class as well
    fitted_orbit, fitted_members = fitter.initial_fit(
        pa.scalar(object_id, type=pa.large_string()), observations
    )
    assert len(fitted_orbit) == 1
    assert fitted_orbit.object_id[0].as_py() == object_id
    assert len(fitted_members) == len(observations)
    # Layup doesn't mark outliers, so everything should be in solution
    assert not np.any(fitted_members.outlier)
    assert np.all(fitted_members.solution)

    assert fitted_orbit.arc_length[0].as_py() > 0
    assert fitted_orbit.num_obs[0].as_py() == len(observations)
    # assert False


def test_not_enough_data(real_data):
    LayupOrbitFitter.bootstrap()
    observations = real_data[:2]
    fitter = LayupOrbitFitter()
    object_id = "2009 JY22"
    fitted_orbit, fitted_members = fitter.initial_fit(object_id, observations)
    assert len(fitted_orbit) == 0
    assert len(fitted_members) == 0
