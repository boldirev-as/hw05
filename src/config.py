def read_config(path):
    if path is None:
        return {}
    out = {}
    with open(path) as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#") or ":" not in line:
                continue
            key, value = line.split(":", 1)
            out[key.strip().replace("-", "_")] = parse_value(value.strip())
    return out


def parse_value(value):
    if value in ("", "null", "None"):
        return None
    if value in ("true", "True"):
        return True
    if value in ("false", "False"):
        return False
    try:
        return int(value)
    except ValueError:
        pass
    try:
        return float(value)
    except ValueError:
        pass
    return value.strip("\"'")
