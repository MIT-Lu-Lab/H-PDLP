import numpy as np
import pytest

from hpdlp import Model, PDLP
from hpdlp._core import solve_once


def pytest_addoption(parser):
    parser.addoption("--require-gpu", action="store_true",
                     help="Fail when the GPU tests cannot run on a CUDA device.")


@pytest.fixture(scope="session")
def cuda_available(request):
    try:
        info = solve_once(np.empty((0, 0)), np.empty(0), 0,
                          None, None, None, None, params={"verbose": False})
    except RuntimeError as error:
        message = str(error)
        unavailable = message.startswith("H-PDLP requires an available CUDA GPU:") and any(
            reason in message.lower() for reason in (
                "no devices found", "no cuda-capable device",
                "cuda driver version is insufficient", "initialization error"))
        if not unavailable or request.config.getoption("--require-gpu"):
            raise
        pytest.skip(message)
    assert info["StatusCode"] == PDLP.OPTIMAL


@pytest.fixture(autouse=True)
def gpu_requirement(request):
    if request.node.get_closest_marker("gpu"):
        request.getfixturevalue("cuda_available")


@pytest.fixture
def base_lp_data():
    return (np.array([1., 1.]), np.array([[1., 2.], [0., 1.], [3., 2.]]),
            np.array([5., -np.inf, -np.inf]), np.array([5., 2., 8.]), None, None)


@pytest.fixture
def model(base_lp_data):
    instance = Model(*base_lp_data)
    instance.setParams(OutputFlag=False, Presolve=False,
                       OptimalityTol=1e-7, FeasibilityTol=1e-7,
                       IterationLimit=20000, TimeLimit=10)
    return instance
