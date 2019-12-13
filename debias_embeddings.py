import numpy as np


def center_matrix(matrix):
    col_avgs = np.average(matrix, axis=0)
    centered = matrix - col_avgs
    assert np.average(centered, axis=0) == np.zeros(matrix.shape[1])

    return matrix - col_avgs


def get_principal_axes(matrix, num_axes=6):
    centered = center_matrix(matrix)
    u, s, v_t = np.linalg.svd(centered)

    return v_t[:num_axes]


def get_bias_subspace(embeddings, gender_def_pairs):
    gender_subspace_matrix = np.zeros(
        (embeddings.shape[1], len(gender_def_pairs) * 2)
    )
    for i, (u, v) in enumerate(gender_def_pairs):
        mean = (u + v) / 2
        gender_subspace_matrix[i*2] = u - mean
        gender_subspace_matrix[(i*2)+1] = v - mean

    return get_principal_axes(gender_subspace_matrix)
