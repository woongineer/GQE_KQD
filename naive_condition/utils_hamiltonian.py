import math
import numpy as np
import pennylane as qml


def _factor_grid(num_qubit: int):
    root = int(math.sqrt(num_qubit))
    for r in range(root, 0, -1):
        if num_qubit % r == 0:
            c = num_qubit // r
            return r, c
    return 1, num_qubit


def _grid_edges(num_qubit: int):
    rows, cols = _factor_grid(num_qubit)

    def idx(r, c):
        return r * cols + c

    edges = []

    for r in range(rows):
        for c in range(cols - 1):
            edges.append((idx(r, c), idx(r, c + 1)))

    for r in range(rows - 1):
        for c in range(cols):
            edges.append((idx(r, c), idx(r + 1, c)))

    return edges


def build_2local_2d_1local_backbone(num_qubit: int):
    coeff_names = []
    ops = []

    edges = _grid_edges(num_qubit)

    for i, j in edges:
        coeff_names.append(f"XX_{i}_{j}")
        ops.append(qml.PauliX(i) @ qml.PauliX(j))

        coeff_names.append(f"YY_{i}_{j}")
        ops.append(qml.PauliY(i) @ qml.PauliY(j))

        coeff_names.append(f"ZZ_{i}_{j}")
        ops.append(qml.PauliZ(i) @ qml.PauliZ(j))

    for q in range(num_qubit):
        coeff_names.append(f"X_{q}")
        ops.append(qml.PauliX(q))

        coeff_names.append(f"Z_{q}")
        ops.append(qml.PauliZ(q))

    return coeff_names, ops


def generate_hamiltonian_family(backbone_ops, num_of_generation, seed, max_coefficient_value):
    rng = np.random.default_rng(seed)
    n_terms = len(backbone_ops)

    family = []
    for _ in range(num_of_generation):
        coeff_vec = rng.integers(
            low=-max_coefficient_value,
            high=max_coefficient_value + 1,
            size=n_terms,
            endpoint=False,
        ).astype(np.int64)

        coeffs = coeff_vec.astype(np.float64).tolist()
        H_op = qml.Hamiltonian(coeffs, backbone_ops)

        family.append((H_op, coeff_vec.copy()))

    return family