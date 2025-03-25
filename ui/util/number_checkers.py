def is_real_number(value: str):
    try:
        float(value)
        return True
    except ValueError:
        return False


def is_positive_real_number(value: str):
    try:
        num = float(value)

        # Check for negative values, infinity, and NaN
        if num < 0 or num in [float("inf"), float("-inf")] or num != num:
            return False
        return True
    except ValueError:
        return False


def is_positive_natural_number(value: str):
    value = value.strip()
    if value.startswith("-") or value.count(".") != 0:
        return False

    if not value.isdigit():
        return False

    try:
        num = int(value)
        return num > 0
    except ValueError:
        return False
