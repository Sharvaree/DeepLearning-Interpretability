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


def get_bias_subspace(embeddings, def_pairs, num_axes=6):
    subspace_basis = np.zeros(
        (embeddings.shape[1], len(def_pairs) * 2)
    )
    for i, (u, v) in enumerate(def_pairs):
        mean = (u + v) / 2
        basis_u = u - mean
        basis_u /= np.linalg.norm(basis_u)
        assert np.linalg.norm(basis_u) == 1
        basis_v = v - mean
        basis_v /= np.linalg.norm(basis_v)
        assert np.linalg.norm(basis_v) == 1

        subspace_basis[i*2] = basis_u
        subspace_basis[(i*2)+1] = basis_v

    return get_principal_axes(subspace_basis, num_axes)


def get_projection(vector, basis_vectors):
    new_vector = np.zeros_like(vector)
    for basis_vector in basis_vectors:
        new_vector += (vector @ basis_vector) @ basis_vector

    return new_vector


def orthogonalize(vector, basis_vectors):
    orthogonal_vector = vector - get_projection(vector, basis_vectors)
    orthogonal_vector /= np.linalg.norm(orthogonal_vector)
    assert np.linalg.norm(orthogonal_vector) == 1

    return orthogonal_vector


def neutralize(embeddings, bias_subspace):
    re_embedded = np.zeros_like(embeddings)
    for i, embedding in enumerate(embeddings):
        # orthogonal_embedding = embedding - get_projection(embedding, bias_subspace)
        # orthogonal_embedding /= np.linalg.norm(orthogonal_embedding)
        # assert np.linalg.norm(orthogonal_embedding) == 1
        # re_embedded[i] = orthogonal_embedding

        re_embedded[i] = orthogonalize(embedding, bias_subspace)

    return re_embedded


def equalize(equalize_pairs, bias_subspace):
    equalized_pairs = np.zeros_like(equalize_pairs)
    for i, (u, v) in enumerate(equalize_pairs):
        mean = orthogonalize(
            (u + v) / 2,
            bias_subspace
        )
        coefficient = np.sqrt(
            1 - np.square(
                np.linalg.norm(mean)
            )
        )
        equalized_u = mean + coefficient * bias_subspace
        equalized_u /= np.linalg.norm(equalized_u)
        assert np.linalg.norm(equalized_u) == 1
        equalized_v = mean - coefficient * bias_subspace
        equalized_v /= np.linalg.norm(equalized_v)
        assert np.linalg.norm(equalized_v) == 1

        equalized_pairs[i*2] = equalized_u
        equalized_pairs[(i*2)+1] = equalized_v

    return equalized_pairs
