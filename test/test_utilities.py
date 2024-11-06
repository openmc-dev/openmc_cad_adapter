import os
import shutil
import difflib
import filecmp

from test import test_config

def diff_files(test_output, gold_file):

    if test_config['update']:
        os.makedirs(os.path.dirname(gold_file), exist_ok=True)
        shutil.copy(test_output, gold_file)

    if not filecmp.cmp(test_output, gold_file):
        print(''.join(difflib.unified_diff(open(gold_file, 'r').readlines(),
                                           open(test_output, 'r').readlines())))
        raise RuntimeError(f'{gold_file} and {test_output} are different')


def diff_gold_file(gold_file, request=None):
    if request is not None:
        gold_file = request.path.parent / gold_file
        b = request.path.parent / 'gold' / gold_file.name
    else:
        b = os.path.join(os.path.dirname(__file__), 'gold', os.path.basename(gold_file))
    diff_files(gold_file, b)