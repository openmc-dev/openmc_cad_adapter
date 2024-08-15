import os
from pathlib import Path

import pytest

from openmc_cad_adapter import to_cubit_journal
import openmc


examples = ["pincell/build_xml.py",
            "lattice/hexagonal/build_xml.py",
            "assembly/assembly.py"]

if 'OPENMC_EXAMPLES_DIR' not in os.environ:
    raise EnvironmentError('Variable OPENMC_EXAMPLES_DIR is required')

OPENMC_EXAMPLES_DIR = Path(os.environ['OPENMC_EXAMPLES_DIR']).resolve()

def example_name(example):
    return '-'.join(example.split('/')[:-1])

@pytest.mark.parametrize("example", examples, ids=example_name)
def test_example(example):

    openmc.reset_auto_ids()
    exec(open(OPENMC_EXAMPLES_DIR / example).read())

    openmc.reset_auto_ids()
    model = openmc.Model.from_xml()

    world = [500, 500, 500]
    to_cubit_journal(model.geometry, world=world, filename=example_name(example))
