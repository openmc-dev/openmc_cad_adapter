import os
import difflib
import filecmp

def diff_files(a, b):
    if not filecmp.cmp(a, b):
        print(''.join(difflib.unified_diff(open(a, 'r').readlines(),
                                           open(b, 'r').readlines())))
        raise RuntimeError(f'{a} and {b} are different')


def diff_gold_file(a, request=None):
    if request is not None:
        a = request.path.parent / a
        b = request.path.parent / 'gold' / a.name
    else:
        b = os.path.join(os.path.dirname(__file__), 'gold', os.path.basename(a))
    diff_files(a, b)