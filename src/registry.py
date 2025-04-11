import flat_bottom

SOLUTIONS = {
    "flat bottom method": flat_bottom.detect,
}


def get_solution(method_name: str):
    return SOLUTIONS[method_name]


def available_solutions():
    return list(SOLUTIONS.keys())


def by_index(index: int):
    if index < 0 or index >= len(SOLUTIONS):
        raise ValueError("Index out of range")
    return list(SOLUTIONS.values())[index]
