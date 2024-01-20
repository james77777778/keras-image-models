import typing


def make_divisible(
    v,
    divisor: int = 8,
    min_value: typing.Optional[float] = None,
    round_limit: float = 0.9,
):
    min_value = min_value or divisor
    new_v = max(min_value, int(v + divisor / 2) // divisor * divisor)
    # Make sure that round down does not go down by more than 10%.
    if new_v < round_limit * v:
        new_v += divisor
    return new_v
