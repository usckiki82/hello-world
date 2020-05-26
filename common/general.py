

def iterable(obj):
    try:
        iter(obj)
    except Exception:
        return False
    else:
        return True