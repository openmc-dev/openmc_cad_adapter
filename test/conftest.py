import pytest

test_config = {'update': False}


def pytest_addoption(parser):
    parser.addoption('--update', action='store_true')


def pytest_configure(config):
    opts = ['update']
    for opt in opts:
        if config.getoption(opt) is not None:
            test_config[opt] = config.getoption(opt)


@pytest.fixture
def run_in_tmpdir(tmpdir):
    orig = tmpdir.chdir()
    try:
        yield
    finally:
        orig.chdir()
