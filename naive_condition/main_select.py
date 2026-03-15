import os
import csv

import numpy as np
import pennylane as qml
import torch
from torch.multiprocessing import Pool
from torch.nn import functional as F

from model import GPT, GPTConfig
from utils_GQE import make_op_pool, apply_circuit, select_token_and_en, temperature
from utils_KQD import (
    make_U,
    make_basis,
    make_S,
    make_Hsub,
    solve_generalized_eigenvalue_problem,
    calc_exact_ground,
    calc_overlap,
    calc_energy_gap,
    calculate_variance,
    calculate_residual_norm,
    ritz_min_and_vector,
)
from utils_general import setup_gpu, setup_logger, save_plt, save_plotly
from utils_hamiltonian import generate_hamiltonian_family, build_2local_2d_1local_backbone

COMPUTE_DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
num_qubit = 8
GPU_id = 0


###########수정된 부분##########
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

    def calculate_loss(self, tokens, energies, prefix_len):
        """
        tokens shape = [B, prefix_len + max_gate]
        prefix는 conditioning only.
        loss는 SEP 뒤 gate token 위치들만 사용한다.
        """
        current_tokens = tokens[:, :-1]
        next_tokens = tokens[:, 1:]

        logits = self(current_tokens)

        next_token_mask = torch.nn.functional.one_hot(
            next_tokens, num_classes=self.config.vocab_size
        )
        next_token_logits = (logits * next_token_mask).sum(dim=2)

        positions = torch.arange(next_tokens.size(1), device=tokens.device)
        gate_pos_mask = (positions >= (prefix_len - 1)).float().unsqueeze(0)

        gated_logits = next_token_logits * gate_pos_mask
        total_logits = torch.sum(gated_logits, dim=1)

        loss = torch.mean(torch.square(total_logits - energies.squeeze()))
        return loss

    @torch.no_grad()
    def generate_from_prefix(
        self,
        prefix_tokens,
        max_new_tokens,
        gate_token_offset,
        temperature=1.0,
        do_sample=True,
    ):
        """
        prefix_tokens: [B, prefix_len]
        여기서부터 gate token만 생성한다.
        coeff token / SEP token은 생성 후보에서 제외한다.
        """
        idx = prefix_tokens.clone()

        for _ in range(max_new_tokens):
            idx_cond = idx if idx.size(1) <= self.config.block_size else idx[:, -self.config.block_size:]
            logits = self(idx_cond)
            logits = logits[:, -1, :]

            # coeff token과 SEP는 생성 금지
            logits[:, :gate_token_offset] = float("inf")

            if do_sample:
                log_probs = F.log_softmax(-logits / temperature, dim=-1)
                probs = log_probs.exp()
                idx_next = torch.multinomial(probs, num_samples=1)
            else:
                idx_next = torch.argmin(logits, dim=-1, keepdim=True)

            idx = torch.cat((idx, idx_next), dim=1)

        return idx


def encode_coeff_vector(coeff_vec, max_coefficient_value):
    """
    coeff in [-C, C] -> token in [0, 2C]
    """
    return (coeff_vec + max_coefficient_value).astype(np.int64)


def decode_gate_tokens_to_ops(token_seq, gate_token_offset, op_pool):
    """
    token_seq shape: [N, max_gate]
    gate token id -> op_pool index로 변환
    """
    op_inds = token_seq - gate_token_offset
    if np.any(op_inds < 0) or np.any(op_inds >= len(op_pool)):
        raise ValueError("gate token decoding failed: token id out of op_pool range.")
    return op_pool[op_inds]


def relative_gap(E, E0, eps=1e-12):
    return float((E - E0) / (abs(E0) + eps))
###########수정된 부분##########


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


###########수정된 부분##########
def compute_ritz_energy_for_sample(sample):
    """
    sample = (ops, H_mat, U_mat, fixed_x, krylov_n)
    """
    ops, H_mat, U_mat, fixed_x, krylov_n = sample

    fixed_x = np.asarray(fixed_x, dtype=np.float64)
    psi = _make_state_from_ops(ops, fixed_x)
    basis = make_basis(U_mat, psi, n=krylov_n)
    S = make_S(basis)
    Hsub = make_Hsub(basis, H_mat)
    evals, _ = solve_generalized_eigenvalue_problem(Hsub, S)
    return float(evals[0])


def get_sequence_energies_kqd(samples, num_workers=8):
    with Pool(processes=num_workers, initializer=_worker_init) as pool:
        energies = pool.map(compute_ritz_energy_for_sample, samples)

    return np.array(energies, dtype=np.float32).reshape(-1, 1)
###########수정된 부분##########


###########수정된 부분##########
def prepare_family_dataset(family_raw, num_qubit, t_evol, max_coefficient_value):
    """
    family_raw: generate_hamiltonian_family(...)의 출력
    반환: list of dict
    """
    dataset = []

    for H_op, coeff_vec in family_raw:
        H_mat = qml.matrix(H_op, wire_order=list(range(num_qubit))).astype(np.complex128)
        U_mat = make_U(H_mat, t=t_evol)
        E0, target_state = calc_exact_ground(H_mat)

        coeff_tokens = encode_coeff_vector(coeff_vec, max_coefficient_value)

        dataset.append({
            "H_op": H_op,
            "H_mat": H_mat,
            "U_mat": U_mat,
            "coeff_vec": coeff_vec,
            "coeff_tokens": coeff_tokens,
            "E0": E0,
            "target_state": target_state,
        })

    return dataset


def build_prefix_tensor(coeff_tokens, sep_token_id, repeat_count, compute_device):
    prefix = np.concatenate([coeff_tokens, np.array([sep_token_id], dtype=np.int64)])
    prefix_batch = np.repeat(prefix[None, :], repeat_count, axis=0)
    return torch.tensor(prefix_batch, dtype=torch.long, device=compute_device)


def generate_and_select_for_one_hamiltonian(
    gpt,
    ham,
    op_pool,
    sep_token_id,
    gate_token_offset,
    max_gate,
    gen_seq_per_hamiltonian,
    train_seq_per_hamiltonian,
    train_temperature,
    krylov_n,
    fixed_x,
    num_workers,
    compute_device,
):
    """
    Hamiltonian 하나에 대해
    1) 여러 sequence 생성
    2) KQD energy 계산
    3) Hamiltonian-wise mu/sigma 계산
    4) local min/max/middle selection
    을 수행한다.
    """
    prefix_batch = build_prefix_tensor(
        coeff_tokens=ham["coeff_tokens"],
        sep_token_id=sep_token_id,
        repeat_count=gen_seq_per_hamiltonian,
        compute_device=compute_device,
    )

    with torch.no_grad():
        full_token_seq_torch = gpt.generate_from_prefix(
            prefix_tokens=prefix_batch,
            max_new_tokens=max_gate,
            gate_token_offset=gate_token_offset,
            temperature=train_temperature,
            do_sample=True,
        )

    full_token_seq = full_token_seq_torch.detach().cpu().numpy()
    prefix_len = len(ham["coeff_tokens"]) + 1
    gate_token_seq = full_token_seq[:, prefix_len:]
    op_seq = decode_gate_tokens_to_ops(gate_token_seq, gate_token_offset, op_pool)

    samples = [(ops, ham["H_mat"], ham["U_mat"], fixed_x, krylov_n) for ops in op_seq]
    seq_en = get_sequence_energies_kqd(samples=samples, num_workers=num_workers)  # [N,1]

    # Hamiltonian-wise local normalization
    mu_h = float(seq_en.mean())
    sigma_h = float(seq_en.std()) + 1e-8
    seq_en_norm = (seq_en - mu_h) / sigma_h

    local_train_size = min(train_seq_per_hamiltonian, len(full_token_seq))
    selected_token_seq, selected_en_norm = select_token_and_en(
        full_token_seq,
        seq_en_norm,
        train_size=local_train_size,
    )

    # local metric 기록용
    best_idx = int(np.argmin(seq_en.squeeze()))
    best_ops = op_seq[best_idx]
    best_E_ritz, _, best_phi = ritz_min_and_vector(
        num_qubit=num_qubit,
        ops=best_ops,
        fixed_x=fixed_x,
        H_mat=ham["H_mat"],
        U_mat=ham["U_mat"],
        n=krylov_n,
    )

    metrics = {
        "ritz_mean": float(seq_en.mean()),
        "best_gap": float(calc_energy_gap(best_E_ritz, ham["E0"])),
        "best_rel_gap": float(relative_gap(best_E_ritz, ham["E0"])),
        "best_overlapK": float(calc_overlap(best_phi, ham["target_state"])),
    }

    return selected_token_seq, selected_en_norm, metrics


def save_test_summary(summary_dict, save_path):
    with open(save_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["metric", "value"])
        for k, v in summary_dict.items():
            writer.writerow([k, v])


def save_nsweep_csv(traces, metric_name, save_path):
    with open(save_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["metric", "hamiltonian_idx", "label", "n", "value"])
        for tr in traces:
            for n_val, y_val in zip(tr["x"], tr["y"]):
                writer.writerow([metric_name, tr["rank"], tr["label"], n_val, y_val])


def evaluate_on_test_family_n_sweep(
    gpt,
    test_family,
    op_pool,
    gate_token_offset,
    sep_token_id,
    max_gate,
    krylov_ns,
    krylov_n_for_report,
    fixed_x,
    compute_device,
    logger,
):
    """
    각 test Hamiltonian마다 sequence를 1개 deterministic하게 생성하고,
    그 sequence에 대해 n-sweep을 수행한다.
    """
    gpt.eval()

    overlap_traces = []
    rel_gap_traces = []

    overlap_last_list = []
    rel_gap_last_list = []
    variance_last_list = []
    residual_last_list = []

    for ham_idx, ham in enumerate(test_family, start=1):
        prefix_batch = build_prefix_tensor(
            coeff_tokens=ham["coeff_tokens"],
            sep_token_id=sep_token_id,
            repeat_count=1,
            compute_device=compute_device,
        )

        with torch.no_grad():
            full_tokens = gpt.generate_from_prefix(
                prefix_tokens=prefix_batch,
                max_new_tokens=max_gate,
                gate_token_offset=gate_token_offset,
                do_sample=False,
            )

        full_tokens_np = full_tokens.detach().cpu().numpy()
        prefix_len = len(ham["coeff_tokens"]) + 1
        gate_token_seq = full_tokens_np[:, prefix_len:]
        op_seq = decode_gate_tokens_to_ops(gate_token_seq, gate_token_offset, op_pool)
        ops = op_seq[0]

        sweep_overlap = []
        sweep_rel_gap = []

        last_var = None
        last_resid = None

        for n in krylov_ns:
            E_ritz, psi0, phi = ritz_min_and_vector(
                num_qubit=num_qubit,
                ops=ops,
                fixed_x=fixed_x,
                H_mat=ham["H_mat"],
                U_mat=ham["U_mat"],
                n=n,
            )
            ovk = calc_overlap(phi, ham["target_state"])
            rgap = relative_gap(E_ritz, ham["E0"])

            sweep_overlap.append(float(ovk))
            sweep_rel_gap.append(float(rgap))

            if n == krylov_n_for_report:
                last_var = float(calculate_variance(phi, ham["H_mat"]))
                last_resid = float(calculate_residual_norm(phi, ham["H_mat"], E=E_ritz))

        overlap_traces.append({
            "rank": ham_idx,
            "label": f"test_ham_{ham_idx}",
            "x": list(krylov_ns),
            "y": sweep_overlap,
        })
        rel_gap_traces.append({
            "rank": ham_idx,
            "label": f"test_ham_{ham_idx}",
            "x": list(krylov_ns),
            "y": sweep_rel_gap,
        })

        overlap_last_list.append(float(sweep_overlap[-1]))
        rel_gap_last_list.append(float(sweep_rel_gap[-1]))
        variance_last_list.append(float(last_var))
        residual_last_list.append(float(last_resid))

        logger.info(
            f"[test {ham_idx}/{len(test_family)}] "
            f"overlapK(n={krylov_ns[-1]})={sweep_overlap[-1]:.6f} | "
            f"rel_gap(n={krylov_ns[-1]})={sweep_rel_gap[-1]:.6f}"
        )

    summary = {
        f"mean_overlapK_at_n_{krylov_ns[-1]}": float(np.mean(overlap_last_list)),
        f"mean_rel_gap_at_n_{krylov_ns[-1]}": float(np.mean(rel_gap_last_list)),
        f"mean_variance_at_n_{krylov_n_for_report}": float(np.mean(variance_last_list)),
        f"mean_residual_at_n_{krylov_n_for_report}": float(np.mean(residual_last_list)),
    }

    return overlap_traces, rel_gap_traces, summary
###########수정된 부분##########


if __name__ == "__main__":
    logger = setup_logger()
    compute_device = setup_gpu(COMPUTE_DEVICE, GPU_id)
    outdir = "results"
    os.makedirs(outdir, exist_ok=True)

    # ===== Hyperparameters =====
    num_workers = 1

    num_feat = 8
    num_scale = 5
    gate_type = ["RX", "RY", "RZ", "CNOT", "H", "I", "MultiRZ"]
    param_scale = np.linspace(start=0.1, stop=1.0, num=num_scale)
    op_pool = make_op_pool(gate_type=gate_type, num_qubit=num_qubit, num_param=num_feat, param_scale=param_scale)
    op_pool_size = len(op_pool)

    max_coefficient_value = 5
    coeff_vocab_size = 2 * max_coefficient_value + 1
    sep_token_id = coeff_vocab_size
    gate_token_offset = coeff_vocab_size + 1
    vocab_size = gate_token_offset + op_pool_size

    coeff_names, backbone_ops = build_2local_2d_1local_backbone(num_qubit)
    coeff_len = len(coeff_names)
    prefix_len = coeff_len + 1  # + SEP

    train_num_hamiltonians = 10   # ex) 500
    test_num_hamiltonians = 2     # ex) 50 or 100
    train_seed = 1234
    test_seed = 5678

    # epoch마다 family subset
    train_hamiltonians_per_epoch = 4

    # Hamiltonian 하나당 먼저 많이 생성한 뒤
    gen_seq_per_hamiltonian = 24   # ex) 256*3에 대응하는 "많이 생성"
    train_seq_per_hamiltonian = 8  # ex) 그중 일부만 selection해서 학습

    n_batches = 4
    max_gate = 28
    max_epoch = 10
    T_max = 100
    T_min = 0.04

    krylov_n = 3
    t_evol = 0.2

    fixed_x = torch.linspace(0.1, 1.0, steps=num_feat).float()

    # test는 deterministic 1-sequence / Hamiltonian
    test_krylov_ns = list(range(krylov_n, krylov_n * 4))
    block_size = prefix_len + max_gate
    ###########수정된 부분##########

    ###########수정된 부분##########
    # ===== Prepare Hamiltonian family =====
    logger.info("Generating train/test Hamiltonian families...")

    train_family_raw = generate_hamiltonian_family(
        backbone_ops=backbone_ops,
        num_of_generation=train_num_hamiltonians,
        seed=train_seed,
        max_coefficient_value=max_coefficient_value,
    )

    test_family_raw = generate_hamiltonian_family(
        backbone_ops=backbone_ops,
        num_of_generation=test_num_hamiltonians,
        seed=test_seed,
        max_coefficient_value=max_coefficient_value,
    )

    train_family = prepare_family_dataset(
        family_raw=train_family_raw,
        num_qubit=num_qubit,
        t_evol=t_evol,
        max_coefficient_value=max_coefficient_value,
    )

    test_family = prepare_family_dataset(
        family_raw=test_family_raw,
        num_qubit=num_qubit,
        t_evol=t_evol,
        max_coefficient_value=max_coefficient_value,
    )

    logger.info(
        f"Prepared family dataset | "
        f"train={len(train_family)}, test={len(test_family)}, "
        f"coeff_len={coeff_len}, prefix_len={prefix_len}, vocab_size={vocab_size}"
    )

    # ===== Load conditioned GQE model =====
    gpt = GPTQE(
        GPTConfig(
            vocab_size=vocab_size,
            block_size=block_size,
            dropout=0.2,
            bias=False,
        )
    ).to(compute_device)

    opt = gpt.configure_optimizers(
        weight_decay=0.01,
        lr=5e-5,
        betas=(0.9, 0.999),
        compute_device=compute_device,
    )
    gpt.train()
    ###########수정된 부분##########

    ###########수정된 부분##########
    loss_history = []
    train_ritz_mean_history = []
    train_best_gap_history = []
    train_best_rel_gap_history = []
    train_best_overlapk_history = []

    # ===== Training =====
    for epoch in range(max_epoch):
        logger.info(f"Starting epoch {epoch + 1}/{max_epoch}...")

        chosen_idx = np.random.choice(
            len(train_family),
            size=min(train_hamiltonians_per_epoch, len(train_family)),
            replace=False,
        )
        selected_family = [train_family[i] for i in chosen_idx]

        selected_token_seq_all = []
        selected_en_norm_all = []

        epoch_ritz_means = []
        epoch_best_gaps = []
        epoch_best_rel_gaps = []
        epoch_best_overlapks = []

        train_temp = temperature(
            T_max=T_max,
            T_min=T_min,
            max_epoch=max_epoch,
            epoch=epoch,
        )

        gpt.eval()
        for ham in selected_family:
            selected_token_seq_h, selected_en_norm_h, local_metrics = generate_and_select_for_one_hamiltonian(
                gpt=gpt,
                ham=ham,
                op_pool=op_pool,
                sep_token_id=sep_token_id,
                gate_token_offset=gate_token_offset,
                max_gate=max_gate,
                gen_seq_per_hamiltonian=gen_seq_per_hamiltonian,
                train_seq_per_hamiltonian=train_seq_per_hamiltonian,
                train_temperature=train_temp,
                krylov_n=krylov_n,
                fixed_x=fixed_x,
                num_workers=num_workers,
                compute_device=compute_device,
            )

            selected_token_seq_all.append(selected_token_seq_h)
            selected_en_norm_all.append(selected_en_norm_h)

            epoch_ritz_means.append(local_metrics["ritz_mean"])
            epoch_best_gaps.append(local_metrics["best_gap"])
            epoch_best_rel_gaps.append(local_metrics["best_rel_gap"])
            epoch_best_overlapks.append(local_metrics["best_overlapK"])
        gpt.train()

        selected_token_seq_all = np.concatenate(selected_token_seq_all, axis=0)
        selected_en_norm_all = np.concatenate(selected_en_norm_all, axis=0)

        tokens = torch.from_numpy(selected_token_seq_all).to(compute_device)
        energies = torch.from_numpy(selected_en_norm_all).to(compute_device)

        token_batches = torch.tensor_split(tokens, n_batches)
        energy_batches = torch.tensor_split(energies, n_batches)

        loss_record = 0.0
        for token_batch, energy_batch in zip(token_batches, energy_batches):
            opt.zero_grad(set_to_none=True)
            loss = gpt.calculate_loss(
                token_batch,
                energy_batch,
                prefix_len=prefix_len,
            )
            loss.backward()
            opt.step()
            loss_record += loss.item() / n_batches

        loss_history.append(loss_record)
        train_ritz_mean_history.append(float(np.mean(epoch_ritz_means)))
        train_best_gap_history.append(float(np.mean(epoch_best_gaps)))
        train_best_rel_gap_history.append(float(np.mean(epoch_best_rel_gaps)))
        train_best_overlapk_history.append(float(np.mean(epoch_best_overlapks)))

        logger.info(
            f"Epoch {epoch + 1} | "
            f"loss={loss_record:.6f} | "
            f"train_RitzE(mean)={train_ritz_mean_history[-1]:.6f} | "
            f"train_best_gap(mean)={train_best_gap_history[-1]:.6f} | "
            f"train_best_rel_gap(mean)={train_best_rel_gap_history[-1]:.6f} | "
            f"train_best_overlapK(mean)={train_best_overlapk_history[-1]:.6f}"
        )
    ###########수정된 부분##########

    logger.info("Training done. Saving training plots...")

    ###########수정된 부분##########
    save_plt(loss_history, outpath=f"{outdir}/loss.png", title="loss", ylabel="loss")
    save_plt(
        train_ritz_mean_history,
        outpath=f"{outdir}/train_ritz_mean.png",
        title="train_RitzE(mean)",
        ylabel="RitzE(mean)",
    )
    save_plt(
        train_best_gap_history,
        outpath=f"{outdir}/train_best_gap_mean.png",
        title="train_best_gap(mean)",
        ylabel="gap",
    )
    save_plt(
        train_best_rel_gap_history,
        outpath=f"{outdir}/train_best_relative_gap_mean.png",
        title="train_best_relative_gap(mean)",
        ylabel="relative gap",
    )
    save_plt(
        train_best_overlapk_history,
        outpath=f"{outdir}/train_best_overlapK_mean.png",
        title="train_best_overlapK(mean)",
        ylabel="overlapK",
    )
    ###########수정된 부분##########

    logger.info("Starting IID test n-sweep evaluation...")

    ###########수정된 부분##########
    overlap_traces, rel_gap_traces, test_summary = evaluate_on_test_family_n_sweep(
        gpt=gpt,
        test_family=test_family,
        op_pool=op_pool,
        gate_token_offset=gate_token_offset,
        sep_token_id=sep_token_id,
        max_gate=max_gate,
        krylov_ns=test_krylov_ns,
        krylov_n_for_report=krylov_n,
        fixed_x=fixed_x,
        compute_device=compute_device,
        logger=logger,
    )

    save_plotly(
        traces=overlap_traces,
        title="Test overlapK vs Krylov n",
        xaxis_title="Krylov n",
        yaxis_title="overlapK",
        save_path=f"{outdir}/test_overlapK_nsweep.html",
    )
    save_plotly(
        traces=rel_gap_traces,
        title="Test relative gap vs Krylov n",
        xaxis_title="Krylov n",
        yaxis_title="relative gap",
        save_path=f"{outdir}/test_relative_gap_nsweep.html",
    )

    save_nsweep_csv(
        traces=overlap_traces,
        metric_name="overlapK",
        save_path=os.path.join(outdir, "test_overlapK_nsweep.csv"),
    )
    save_nsweep_csv(
        traces=rel_gap_traces,
        metric_name="relative_gap",
        save_path=os.path.join(outdir, "test_relative_gap_nsweep.csv"),
    )
    save_test_summary(
        test_summary,
        save_path=os.path.join(outdir, "test_summary.csv"),
    )

    logger.info("=== Final IID test summary ===")
    for k, v in test_summary.items():
        logger.info(f"{k}: {v:.6f}")

    logger.info("Finished.")
    ###########수정된 부분##########