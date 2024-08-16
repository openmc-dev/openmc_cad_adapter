import os
import difflib
import filecmp
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


def diff_files(a, b):
    if not filecmp.cmp(a, b):
        print(''.join(difflib.unified_diff(open(a, 'r').readlines(),
                                           open(b, 'r').readlines())))
        raise RuntimeError(f'{a} and {b} are different')


def example_name(example):
    return '-'.join(example.split('/')[:-1])


@pytest.mark.parametrize("example", examples, ids=example_name)
def test_examples(example, request):

    openmc.reset_auto_ids()
    exec(open(OPENMC_EXAMPLES_DIR / example).read())

    openmc.reset_auto_ids()
    model = openmc.Model.from_xml()

    world = [500, 500, 500]
    output = example_name(example)
    to_cubit_journal(model.geometry, world=world, filename=output)

    gold_file = request.path.parent / Path('gold') / Path(output)
    diff_files(output, gold_file)
