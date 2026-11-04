import pytest

import morphjax  # noqa: F401  (enables float64)
from morphjax.aero import build_aero_model
from morphjax.data import load_aero_tables
from morphjax.dynamics import Plant


@pytest.fixture(scope="session")
def tables():
    return load_aero_tables()


@pytest.fixture(scope="session")
def plant(tables):
    return Plant(build_aero_model(tables))


@pytest.fixture(scope="session")
def setup():
    from morphjax import pipeline

    return pipeline.build()
