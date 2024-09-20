_CUBIT_ID = 1


def reset_cubit_ids():
    global _CUBIT_ID
    _CUBIT_ID = 1

def lastid():
    global _CUBIT_ID
    id_out = _CUBIT_ID
    _CUBIT_ID += 1
    return id_out


def emit_get_last_id(type="body", cmds=None):
    idn = lastid()
    ids = f"id{idn}"
    if cmds is not None:
        cmds.append(f'#{{ {ids} = Id("{type}") }}')
    else:
        print('Warning: cmds is None')
    return ids
