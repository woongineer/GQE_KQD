import numpy as np
import pennylane as qml


def make_op_pool(gate_type, num_qubit, num_param, param_scale):
    op_pool = []
    for gate in gate_type:
        if gate in ["RX", "RY", "RZ"]:
            for q in range(num_qubit):
                for p in range(num_param):
                    for s in param_scale:
                        op_pool.append((gate, (p, s), (q, None)))
        elif gate in ["H", "I"]:
            for q in range(num_qubit):
                op_pool.append((gate, None, (q, None)))
        elif gate == "CNOT":
            for control in range(num_qubit):
                for target in range(num_qubit):
                    if control != target:
                        op_pool.append((gate, None, (control, target)))
        elif gate == "MultiRZ":
            for q1 in range(num_qubit):
                for q2 in range(q1 + 1, num_qubit):
                    for p in range(num_param):
                        for s in param_scale:
                            op_pool.append((gate, (p, s), (q1, q2)))
    return np.array(op_pool, dtype=object)


def apply_circuit(x, circuit):
    for gate in circuit:
        apply_gate(gate, x)


def apply_gate(gate, x):
    gate_type, param_idx, qubit_idx = gate
    ctrl_idx, target_idx = qubit_idx
    if gate_type == "RX":
        qml.RX(x[param_idx[0]] * param_idx[1], wires=ctrl_idx)
    elif gate_type == "RY":
        qml.RY(x[param_idx[0]] * param_idx[1], wires=ctrl_idx)
    elif gate_type == "RZ":
        qml.RZ(x[param_idx[0]] * param_idx[1], wires=ctrl_idx)
    elif gate_type == "H":
        qml.Hadamard(wires=ctrl_idx)
    elif gate_type == "CNOT":
        qml.CNOT(wires=[ctrl_idx, target_idx])
    elif gate_type == "I":
        qml.Identity(wires=ctrl_idx)
    elif gate_type == "MultiRZ":
        qml.MultiRZ(x[param_idx[0]] * param_idx[1], wires=[ctrl_idx, target_idx])


def select_token_and_en(train_token_seq, train_seq_en, train_size):
    k = int(train_size * 0.4)
    middle = train_size - (k * 2)

    sorted_indices = np.argsort(train_seq_en[:, -1])

    top_indices = sorted_indices[:k]
    bottom_indices = sorted_indices[-k:]

    middle_pool_indices = sorted_indices[k:-k]
    interval_points = np.linspace(0, len(middle_pool_indices) - 1, num=middle)
    middle_sample_indices_in_pool = np.round(interval_points).astype(int)
    middle_indices = middle_pool_indices[middle_sample_indices_in_pool]

    final_indices = np.concatenate([top_indices, bottom_indices, middle_indices])
    np.random.shuffle(final_indices)

    new_train_token_seq = train_token_seq[final_indices]
    new_train_seq_en = train_seq_en[final_indices]

    return new_train_token_seq, new_train_seq_en


def normalize_E(E, mu, sigma):
    return (E - mu) / sigma


def temperature(T_max, T_min, max_epoch, epoch):
    ratio = (T_min / T_max) ** (epoch / max_epoch)
    return T_max * ratio
