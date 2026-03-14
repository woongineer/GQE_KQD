import os
from functools import partial

import numpy as np
import pennylane as qml
import torch
from torch.multiprocessing import Pool
from torch.nn import functional as F

from model import GPT, GPTConfig
from utils_GQE import make_op_pool, apply_circuit, select_token_and_en, normalize_E, temperature
from utils_KQD import make_U, make_basis, make_S, make_Hsub, solve_generalized_eigenvalue_problem, calc_exact_ground, \
    calculate_overlap, calculate_energy_gap, calculate_variance, calculate_residual_norm, ritz_min_and_vector, \
    select_global_top_k_candidates, run_n_sweep_for_candidates
from utils_general import setup_gpu, setup_logger, save_plt, save_plotly, save_csv

COMPUTE_DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
num_qubit = 8
GPU_id = 0


class GPTQE(GPT):
    def forward(self, idx):
        compute_device = idx.device
        b, t = idx.size()
        pos = torch.arange(0, t, dtype=torch.long, device=compute_device)
        tok_emb = self.transformer.wte(idx)
        pos_emb = self.transformer.wpe(pos)
        x = self.transformer.drop(tok_emb + pos_emb)
        for block in self.transformer.h:
            x = block(x)
        x = self.transformer.ln_f(x)
        logits = self.lm_head(x)
        return logits

    def calculate_loss(self, tokens, energies):
        current_tokens, next_tokens = tokens[:, :-1], tokens[:, 1:]
        logits = self(current_tokens)
        next_token_mask = torch.nn.functional.one_hot(next_tokens, num_classes=self.config.vocab_size)
        next_token_logits = (logits * next_token_mask).sum(axis=2)
        total_logits = torch.sum(next_token_logits, dim=1)
        loss = torch.mean(torch.square(total_logits - energies.squeeze()))
        return loss

    @torch.no_grad()
    def generate(self, n_sequences, max_new_tokens, temperature, compute_device):
        idx = torch.zeros(size=(n_sequences, 1), dtype=torch.long, device=compute_device)
        total_energy = torch.zeros((n_sequences, 1), device=compute_device)

        for _ in range(max_new_tokens):
            idx_cond = idx if idx.size(1) <= self.config.block_size else idx[:, -self.config.block_size:]
            logits = self(idx_cond)
            logits = logits[:, -1, :]
            logits[:, 0] = float("inf")

            log_probs = F.log_softmax(-logits / temperature, dim=-1)
            probs = log_probs.exp()

            idx_next = torch.multinomial(probs, num_samples=1)
            total_energy += torch.gather(logits, 1, idx_next)
            idx = torch.cat((idx, idx_next), dim=1)

        return idx, total_energy


# Problem Hamiltonian
def make_H_3_local_ata(num_qubit):
    """
    B-style fixed (non-random):
      - all-to-all 2-local: for all i<j add XX,YY,ZZ
      - + some 3-local mixed-Pauli terms (local triples)
      - + small 1-local fields to break symmetry
    """
    ops, coeffs = [], []

    # 2-local all-to-all
    pairs = [(i, j) for i in range(num_qubit) for j in range(i + 1, num_qubit)]
    for idx, (i, j) in enumerate(pairs):
        Jxx = float(7 + (3 * idx) % 17)
        Jyy = float(5 + (5 * idx) % 19)
        Jzz = float(3 + (7 * idx) % 23)
        ops.append(qml.PauliX(i) @ qml.PauliX(j)); coeffs.append(Jxx)
        ops.append(qml.PauliY(i) @ qml.PauliY(j)); coeffs.append(Jyy)
        ops.append(qml.PauliZ(i) @ qml.PauliZ(j)); coeffs.append(Jzz)

    # 1-local fields (light)
    for i in range(num_qubit):
        if i % 2 == 0:
            ops.append(qml.PauliX(i)); coeffs.append(4.0 + (i % 3))
        if i % 3 == 0:
            ops.append(qml.PauliZ(i)); coeffs.append(6.0 + (i % 4))

    # 3-local terms: only consecutive triples to keep term count reasonable
    triples = [(i, i + 1, i + 2) for i in range(num_qubit - 2)]
    pauli_triple = [
        (qml.PauliX, qml.PauliY, qml.PauliZ),
        (qml.PauliX, qml.PauliZ, qml.PauliY),
        (qml.PauliY, qml.PauliX, qml.PauliZ),
        (qml.PauliZ, qml.PauliX, qml.PauliY),
    ]
    for t, (i, j, k) in enumerate(triples):
        P1, P2, P3 = pauli_triple[t % 4]
        ops.append(P1(i) @ P2(j) @ P3(k))
        coeffs.append(float(21 + (2 * t) % 11))

    H_op = qml.Hamiltonian(coeffs, ops)
    H_mat = qml.matrix(H_op, wire_order=list(range(num_qubit))).astype(np.complex128)
    return H_op, H_mat


# ---- multiprocessing worker ----
quantum_device = None

def _worker_init():
    global quantum_device
    quantum_device = qml.device("lightning.qubit", wires=num_qubit)


def _make_state_from_ops(ops, fixed_x):
    @qml.qnode(quantum_device, interface="autograd")
    def run_state():
        apply_circuit(fixed_x, ops)
        return qml.state()

    psi = run_state()
    return np.asarray(psi, dtype=np.complex128)


def compute_ritz_energy_for_ops(ops, krylov_n, H_mat, U_mat, fixed_x):
    fixed_x = np.asarray(fixed_x, dtype=np.float64)

    psi = _make_state_from_ops(ops, fixed_x)
    basis = make_basis(U_mat, psi, n=krylov_n)
    S = make_S(basis)
    Hsub = make_Hsub(basis, H_mat)
    evals, _ = solve_generalized_eigenvalue_problem(Hsub, S)
    return float(evals[0])


def get_sequence_energies_kqd(op_seq, num_workers=8, krylov_n=3, H_mat=None, U_mat=None, fixed_x=None):
    worker_func = partial(compute_ritz_energy_for_ops, krylov_n=krylov_n, H_mat=H_mat, U_mat=U_mat, fixed_x=fixed_x)

    with Pool(processes=num_workers, initializer=_worker_init) as pool:
        energies = pool.map(worker_func, list(op_seq))

    return np.array(energies, dtype=np.float32).reshape(-1, 1)


if __name__ == "__main__":
    logger = setup_logger()
    compute_device = setup_gpu(COMPUTE_DEVICE, GPU_id)
    outdir = "GQE_KQD_results"
    os.makedirs(outdir, exist_ok=True)

    # ===== Hyperparameters(workers, GQE, Training, KQD, X, evaluation samples k) =====
    num_workers = 16  # 8

    num_feat = 8
    num_scale = 5
    gate_type = ["RX", "RY", "RZ", "CNOT", "H", "I", "MultiRZ"]
    param_scale = np.linspace(start=0.1, stop=1.0, num=num_scale)
    op_pool = make_op_pool(gate_type=gate_type, num_qubit=num_qubit, num_param=num_feat, param_scale=param_scale)
    op_pool_size = len(op_pool)

    train_size = 256  # 256
    n_batches = 4
    max_gate = 28
    max_epoch = 10  # 2000
    T_max = 100
    T_min = 0.04

    krylov_n = 3
    t_evol = 0.2

    fixed_x = torch.linspace(0.1, 1.0, steps=num_feat).float()

    top_k = 10

    # ===== Preparing Hamiltonian =====
    H_op, H_mat = make_H_3_local_ata(num_qubit)
    U_mat = make_U(H_mat, t=t_evol)

    # exact ground (target)
    E0, target_state = calc_exact_ground(H_mat)
    logger.info(f"[exact] Ground energy E0 = {E0:.6f}")

    # ===== Loading GQE model =====
    gpt = GPTQE(GPTConfig(vocab_size=op_pool_size + 1, block_size=max_gate, dropout=0.2, bias=False)).to(compute_device)
    opt = gpt.configure_optimizers(weight_decay=0.01, lr=5e-5, betas=(0.9, 0.999), compute_device=compute_device)
    gpt.train()

    mu, sigma = None, None

    loss_history = []
    ritz_mean_history = []
    ritz_top_k_history = []
    overlap0_top_k_history = []
    overlapk_top_k_history = []
    gap_top_k_history = []
    var_top_k_history = []
    resid_top_k_history = []
    candidate_ops = []

    # ===== Training =====
    for i in range(max_epoch):
        logger.info(f"Starting epoch {i + 1}/{max_epoch}...")

        gpt.eval()
        train_token_seq_torch, _ = gpt.generate(
            n_sequences=train_size * 3,
            max_new_tokens=max_gate,
            temperature=temperature(T_max=T_max, T_min=T_min, max_epoch=max_epoch, epoch=i),
            compute_device=compute_device,
        )
        gpt.train()

        train_token_seq = train_token_seq_torch.detach().cpu().numpy()
        train_op_inds = train_token_seq[:, 1:] - 1
        train_op_seq = op_pool[train_op_inds]

        train_seq_en = get_sequence_energies_kqd(
            train_op_seq,
            num_workers=num_workers,
            krylov_n=krylov_n,
            H_mat=H_mat,
            U_mat=U_mat,
            fixed_x=fixed_x,
        )

        alpha = 0.1
        if mu is None:
            mu, sigma = float(train_seq_en.mean()), float(train_seq_en.std()) + 1e-8
        else:
            mu = alpha * float(train_seq_en.mean()) + (1 - alpha) * mu
            sigma = alpha * float(train_seq_en.std()) + (1 - alpha) * sigma

        train_seq_en_norm = normalize_E(train_seq_en, mu, sigma)
        train_token_seq, train_seq_en_norm = select_token_and_en(train_token_seq, train_seq_en_norm, train_size)

        tokens = torch.from_numpy(train_token_seq).to(compute_device)
        energies = torch.from_numpy(train_seq_en_norm).to(compute_device)

        train_inds = np.arange(train_size)
        token_batches = torch.tensor_split(tokens[train_inds], n_batches)
        energy_batches = torch.tensor_split(energies[train_inds], n_batches)

        loss_record = 0.0
        for token_batch, energy_batch in zip(token_batches, energy_batches):
            opt.zero_grad(set_to_none=True)
            loss = gpt.calculate_loss(token_batch, energy_batch)
            loss.backward()
            opt.step()
            loss_record += loss.item() / n_batches

        loss_history.append(loss_record)

        sort_idx = np.argsort(train_seq_en.squeeze())
        top_k_idx = sort_idx[:top_k]
        top_k_ops = [train_op_seq[j] for j in top_k_idx]

        # ===== Mid-Training Evaluation =====
        top_k_metrics = []
        for ops in top_k_ops:
            E_ritz, psi0, phi = ritz_min_and_vector(num_qubit, ops, fixed_x, H_mat, U_mat, n=krylov_n)
            ov0 = calculate_overlap(psi0, target_state)
            ovk = calculate_overlap(phi, target_state)
            gap = calculate_energy_gap(E_ritz, E0)
            var = calculate_variance(phi, H_mat)
            resid = calculate_residual_norm(phi, H_mat, E=E_ritz)

            top_k_metrics.append((E_ritz, ov0, ovk, gap, var, resid, ops))

            candidate_ops.append({"epoch": i + 1, "ops": ops, "energy": float(E_ritz)})

        ritz_mean_history.append(float(train_seq_en.mean()))
        ritz_top_k_history.append(float(np.mean([m[0] for m in top_k_metrics])))
        overlap0_top_k_history.append(float(np.mean([m[1] for m in top_k_metrics])))
        overlapk_top_k_history.append(float(np.mean([m[2] for m in top_k_metrics])))
        gap_top_k_history.append(float(np.mean([m[3] for m in top_k_metrics])))
        var_top_k_history.append(float(np.mean([m[4] for m in top_k_metrics])))
        resid_top_k_history.append(float(np.mean([m[5] for m in top_k_metrics])))

        logger.info(f"Iter {i + 1} | loss={loss_record:.6f} | RitzE(mean)={float(train_seq_en.mean()):.6f}")

    logger.info("Training Done, Evaluation starting...")


    # ===== Save epochwise plots =====
    save_plt(loss_history, outpath=f"{outdir}/loss.png", title="loss", ylabel="loss")
    save_plt(ritz_mean_history, outpath=f"{outdir}/ritz_mean.png", title="RitzE(mean)", ylabel="RitzE(mean)")
    save_plt(ritz_top_k_history, outpath=f"{outdir}/ritz_topk.png", title=f"RitzE(top_{top_k})", ylabel="RitzE(topk)")
    save_plt(overlap0_top_k_history, outpath=f"{outdir}/overlap0_topk.png", title=f"overlap0(top_{top_k})", ylabel="overlap0(topk)")
    save_plt(overlapk_top_k_history, outpath=f"{outdir}/overlapk_topk.png", title=f"overlapk(top_{top_k})", ylabel="overlapk(topk)")
    save_plt(gap_top_k_history, outpath=f"{outdir}/gap_topk.png", title=f"gap(top_{top_k})", ylabel="gap(topk)")
    save_plt(var_top_k_history, outpath=f"{outdir}/var_topk.png", title=f"var(top_{top_k})", ylabel="var(topk)")
    save_plt(resid_top_k_history, outpath=f"{outdir}/resid_topk.png", title=f"resid(top_{top_k})", ylabel="resid(topk)")

    # ===== Evaluation =====
    ns = list(range(krylov_n, krylov_n * 4))
    global_top_candidates = select_global_top_k_candidates(candidate_ops, top_k=top_k)

    nsweep_results = run_n_sweep_for_candidates(top_candidates=global_top_candidates,
                                                ns=ns, num_qubit=num_qubit, fixed_x=fixed_x, H_mat=H_mat, U_mat=U_mat,
                                                target_state=target_state, E0=E0)

    save_plotly(traces=nsweep_results["RitzE"],
                title=f"RitzE (top_{top_k})", xaxis_title="krylov n", yaxis_title="RitzE",
                save_path=f"{outdir}/n_sweep_top_{top_k}_RitzE.html")
    save_plotly(traces=nsweep_results["overlapK"],
                title=f"overlapK (top_{top_k})", xaxis_title="krylov n", yaxis_title="overlapK",
                save_path=f"{outdir}/n_sweep_top_{top_k}_overlapK.html")
    save_plotly(traces=nsweep_results["gap"],
                title=f"gap (top_{top_k})", xaxis_title="krylov n", yaxis_title="gap",
                save_path=f"{outdir}/n_sweep_top_{top_k}_gap.html")
    save_plotly(traces=nsweep_results["variance"],
                title=f"variance (top_{top_k})", xaxis_title="krylov n", yaxis_title="variance",
                save_path=f"{outdir}/n_sweep_top_{top_k}_variance.html")
    save_plotly(traces=nsweep_results["residual"],
                title=f"residual (top_{top_k})", xaxis_title="krylov n", yaxis_title="residual",
                save_path=f"{outdir}/n_sweep_top_{top_k}_residual.html")

    save_csv(nsweep_results=nsweep_results, save_path=os.path.join(outdir, "nsweep_topk_all_metrics.csv"))

    logger.info("Finished")
