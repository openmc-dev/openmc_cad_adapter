import os
import difflib
import filecmp
from pathlib import Path

import subprocess

import pytest

from openmc_cad_adapter import to_cubit_journal
import openmc

from test import diff_files


examples = ["pincell/build_xml.py",
            "lattice/hexagonal/build_xml.py",
            "assembly/assembly.py"]


if 'OPENMC_EXAMPLES_DIR' not in os.environ:
    raise EnvironmentError('Variable OPENMC_EXAMPLES_DIR is required')


OPENMC_EXAMPLES_DIR = Path(os.environ['OPENMC_EXAMPLES_DIR']).resolve()


def example_name(example):
    return '-'.join(example.split('/')[:-1])


def generate_example_xml(example):
    if 'assembly' in example:
        subprocess.Popen(['python', str(OPENMC_EXAMPLES_DIR / example), '--generate']).wait()
    else:
        exec(open(OPENMC_EXAMPLES_DIR / example).read())


@pytest.mark.parametrize("example", examples, ids=example_name)
def test_examples(example, request, run_in_tmpdir):

    openmc.reset_auto_ids()
    generate_example_xml(example)

    openmc.reset_auto_ids()
    model = openmc.Model.from_xml()

    world = [500, 500, 500]
    output = example_name(example) + '.jou'
    to_cubit_journal(model.geometry, world=world, filename=output)

    gold_file = request.path.parent / Path('gold') / Path(output)
    diff_files(output, gold_file)


def test_cell_by_cell_conversion(request, run_in_tmpdir):
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
def test_examples_cli(example, request, run_in_tmpdir):

    openmc.reset_auto_ids()
    generate_example_xml(example)

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
