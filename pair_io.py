def get_pairs(filepath):
    with open(filepath, 'r') as f:
        lines = f.readlines()
    lines = [line.strip().lower() for line in lines]
    return [line.split() for line in lines]


def get_definitional_pairs(filepath='./resources/definitional_pairs.txt'):
    return get_pairs(filepath)


def get_equalization_pairs(filepath='./resources/equalize_pairs.txt'):
    return get_pairs(filepath)
