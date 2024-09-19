import os
import difflib
import filecmp
from pathlib import Path

import subprocess

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
    output = example_name(example) + '.jou'
    to_cubit_journal(model.geometry, world=world, filename=output)

    gold_file = request.path.parent / Path('gold') / Path(output)
    diff_files(output, gold_file)


def test_cell_by_cell_conversion(request):
    openmc.reset_auto_ids()
    exec(open(OPENMC_EXAMPLES_DIR / "pincell/build_xml.py").read())

    openmc.reset_auto_ids()
    model = openmc.Model.from_xml()

    cell_ids = list(model.geometry.get_all_cells().keys())

    world = [500, 500, 500]
    output = 'pincell'
    to_cubit_journal(model.geometry, world=world, cells=cell_ids, filename=output)

    for cell_id in cell_ids:
        output = f'pincell_cell{cell_id}.jou'
        gold_file = request.path.parent / Path('gold') / Path(output)
        diff_files(output, gold_file)

@pytest.mark.parametrize("example", examples, ids=example_name)
def test_examples_cli(example, request):

    openmc.reset_auto_ids()
    example_path = OPENMC_EXAMPLES_DIR / example
    exec(open(example_path).read())

    openmc.reset_auto_ids()
    world = [500, 500, 500]
    output = example_name(example) + '.jou'
    cmd = ['openmc_to_cad', '.', '-o', output, '--world'] + [str(w) for w in world]
    pipe = subprocess.Popen(cmd)
    pipe.wait()
    if pipe.returncode != 0:
        raise RuntimeError(f'Command {" ".join(cmd)} failed')

    gold_file = request.path.parent / Path('gold') / Path(output)
    diff_files(output, gold_file)
