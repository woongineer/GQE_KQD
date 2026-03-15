import numpy as np
import pennylane as qml

from utils_GQE import apply_circuit


def make_U(H_mat, t):
    """U = exp(-i H t)"""
    evals, evecs = np.linalg.eigh(H_mat)
    phase = np.exp(-1j * evals * t)
    U = (evecs * phase) @ evecs.conj().T
    return U


def make_phi(k, U, psi):
    """phi_k = U^k psi"""
    out = np.asarray(psi, dtype=np.complex128)
    for _ in range(k):
        out = U @ out
    return out


def make_basis(U, psi, n=3):
    return [make_phi(k, U, psi) for k in range(n)]


def make_S(basis):
    n = len(basis)
    S = np.zeros((n, n), dtype=np.complex128)
    for i in range(n):
        for j in range(n):
            S[i, j] = np.vdot(basis[i], basis[j])
    return S


def make_Hsub(basis, H_mat):
    n = len(basis)
    Hsub = np.zeros((n, n), dtype=np.complex128)
    for i in range(n):
        for j in range(n):
            Hsub[i, j] = np.vdot(basis[i], H_mat @ basis[j])
    return Hsub


def ritz_min_and_vector(num_qubit, ops, fixed_x, H_mat, U_mat, n=3, dev=None):
    """
    ops -> psi -> basis -> (Hsub,S) -> GEP -> min Ritz value + Ritz vector(원래 공간)
    return:
      E_ritz: float
      psi: initial state
      phi_ritz: Ritz vector in full space
    """
    dev_eval = dev if dev is not None else qml.device("lightning.qubit", wires=num_qubit)

    @qml.qnode(dev_eval, interface="autograd")
    def _state():
        apply_circuit(np.asarray(fixed_x, dtype=np.float64), ops)
        return qml.state()

    psi = np.asarray(_state(), dtype=np.complex128)
    basis = make_basis(U_mat, psi, n=n)
    S = make_S(basis)
    Hsub = make_Hsub(basis, H_mat)
    evals, Y = solve_generalized_eigenvalue_problem(Hsub, S)

    idx = int(np.argmin(evals))
    E_ritz = float(evals[idx])

    y = Y[:, idx]  # coefficients
    phi = np.zeros_like(basis[0], dtype=np.complex128)
    for i in range(n):
        phi += y[i] * basis[i]
    phi = phi / np.linalg.norm(phi)

    psi = psi / np.linalg.norm(psi)
    return E_ritz, psi, phi


def solve_generalized_eigenvalue_problem(Hsub, S):
    """
    Hsub y = λ S y
    return:
      evals: (n,) 오름차순
      Y: (n,n) 각 열이 y 벡터 (원 문제의 계수)
    """
    L = np.linalg.cholesky(S)  # S = L L^†
    Linv = np.linalg.inv(L)
    A = Linv @ Hsub @ Linv.conj().T  # A z = λ z
    evals, Z = np.linalg.eigh(A)  # Z columns are z
    # y = L^{-†} z
    Y = np.linalg.inv(L.conj().T) @ Z
    return evals.real, Y


def calc_exact_ground(H_mat):
    """
    exact diagonalization (4 qubit=16차원이라 그냥 가능)
    return:
      E0: ground energy (float)
      g: ground state vector (complex, shape (16,))
    """
    evals, evecs = np.linalg.eigh(H_mat)
    idx = int(np.argmin(evals.real))
    E0 = float(evals[idx].real)
    g = evecs[:, idx].astype(np.complex128)
    # normalize (안전)
    g = g / np.linalg.norm(g)
    return E0, g


def calc_overlap(state_a, state_b):
    """
    |<a|b>|^2
    """
    a = np.asarray(state_a, dtype=np.complex128)
    b = np.asarray(state_b, dtype=np.complex128)
    a = a / np.linalg.norm(a)
    b = b / np.linalg.norm(b)
    return float(np.abs(np.vdot(a, b)) ** 2)


def calculate_energy(state, H_mat):
    """
    <psi|H|psi>
    """
    psi = np.asarray(state, dtype=np.complex128)
    psi = psi / np.linalg.norm(psi)
    return float(np.real(np.vdot(psi, H_mat @ psi)))


def calc_energy_gap(E, E0):
    return float(E - E0)


def calc_relative_gap(E, E0, eps=1e-12):
    return float((E - E0) / (abs(E0) + eps))


def calculate_variance(state, H_mat):
    """
    Var_H(psi) = <H^2> - <H>^2
    """
    psi = np.asarray(state, dtype=np.complex128)
    psi = psi / np.linalg.norm(psi)
    Hpsi = H_mat @ psi
    E = np.real(np.vdot(psi, Hpsi))
    H2psi = H_mat @ Hpsi
    E2 = np.real(np.vdot(psi, H2psi))
    return float(E2 - E * E)


def calculate_residual_norm(state, H_mat, E=None):
    """
    || H|psi> - E|psi> ||
    E를 안 주면 E=<psi|H|psi>로 사용
    """
    psi = np.asarray(state, dtype=np.complex128)
    psi = psi / np.linalg.norm(psi)
    Hpsi = H_mat @ psi
    if E is None:
        E = np.real(np.vdot(psi, Hpsi))
    r = Hpsi - E * psi
    return float(np.linalg.norm(r))


def select_global_top_k_candidates(candidate_ops, top_k):
    sorted_candidates = sorted(candidate_ops, key=lambda x: x["energy"])

    unique_candidates = []
    seen_ops = set()

    for cand in sorted_candidates:
        ops_key = tuple(str(op) for op in cand["ops"])
        if ops_key in seen_ops:
            continue

        seen_ops.add(ops_key)
        unique_candidates.append(cand)

        if len(unique_candidates) >= top_k:
            break

    return unique_candidates


def run_n_sweep_for_candidates(top_candidates, ns, num_qubit, fixed_x, H_mat, U_mat, target_state, E0):
    results = {"RitzE": [], "overlap0": [], "overlapK": [], "gap": [], "variance": [], "residual": []}

    for rank, cand in enumerate(top_candidates, start=1):
        ops = cand["ops"]
        epoch = cand["epoch"]
        label = f"rank_{rank}_epoch_{epoch}"

        sweep_E = []
        sweep_ovK = []
        sweep_gap = []
        sweep_var = []
        sweep_resid = []

        for n in ns:
            E_ritz, psi0, phi = ritz_min_and_vector(num_qubit, ops, fixed_x, H_mat, U_mat, n=n)
            sweep_E.append(float(E_ritz))
            sweep_ovK.append(float(calc_overlap(phi, target_state)))
            sweep_gap.append(float(calc_energy_gap(E_ritz, E0)))
            sweep_var.append(float(calculate_variance(phi, H_mat)))
            sweep_resid.append(float(calculate_residual_norm(phi, H_mat, E=E_ritz)))

        results["RitzE"].append({"rank": rank, "epoch": epoch, "label": label, "x": list(ns), "y": sweep_E})
        results["overlapK"].append({"rank": rank,"epoch": epoch,"label": label,"x": list(ns),"y": sweep_ovK})
        results["gap"].append({"rank": rank,"epoch": epoch,"label": label,"x": list(ns),"y": sweep_gap})
        results["variance"].append({"rank": rank,"epoch": epoch,"label": label,"x": list(ns),"y": sweep_var})
        results["residual"].append({"rank": rank,"epoch": epoch,"label": label,"x": list(ns),"y": sweep_resid})

    return results