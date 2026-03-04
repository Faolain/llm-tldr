def helper(value):
    return value + 1


def use_helper(seed):
    temp = helper(seed)
    return temp * 2


def track_flow(x):
    origin = x + 1
    derived = origin * 3
    return derived
