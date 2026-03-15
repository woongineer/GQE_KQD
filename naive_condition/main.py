import os

import numpy as np
import pennylane as qml
import torch
from torch.multiprocessing import Pool
from torch.nn import functional as F

from model import GPT, GPTConfig
from utils_GQE import make_op_pool, temperature
from utils_KQD import make_U, calc_exact_ground, calc_overlap, calc_energy_gap, ritz_min_and_vector, calc_relative_gap
from utils_general import setup_gpu, setup_logger, save_plt, save_plotly
from utils_hamiltonian import generate_hamiltonian_family, build_2local_2d_1local_backbone

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

    def calculate_loss(self, tokens, energies, prefix_len):
        current_tokens, next_tokens = tokens[:, :-1], tokens[:, 1:]
        logits = self(current_tokens)

        next_token_mask = F.one_hot(next_tokens, num_classes=self.config.vocab_size)
        next_token_logits = (logits * next_token_mask).sum(axis=2)

        positions = torch.arange(next_tokens.size(1), device=tokens.device)
        gate_pos_mask = (positions >= (prefix_len - 1)).float().unsqueeze(0)

        total_logits = torch.sum(next_token_logits * gate_pos_mask, dim=1)
        loss = torch.mean(torch.square(total_logits - energies.squeeze()))
        return loss

    @torch.no_grad()
    def generate_from_prefix(self, prefix_tokens, max_new_tokens, temperature, gate_token_offset):
        idx = prefix_tokens.clone()

        for _ in range(max_new_tokens):
            idx_cond = idx if idx.size(1) <= self.config.block_size else idx[:, -self.config.block_size:]
            logits = self(idx_cond)
            logits = logits[:, -1, :]
            logits[:, :gate_token_offset] = float("inf")

            log_probs = F.log_softmax(-logits / temperature, dim=-1)
            probs = log_probs.exp()

            idx_next = torch.multinomial(probs, num_samples=1)
            idx = torch.cat((idx, idx_next), dim=1)

        return idx


def encode_coeff_vector(coeff_vec, max_coefficient_value):
    return (coeff_vec + max_coefficient_value).astype(np.int64)


# ---- multiprocessing worker ----
quantum_device = None


def _worker_init():
    global quantum_device
    quantum_device = qml.device("lightning.qubit", wires=num_qubit)


def compute_ritz_energy_for_sample(sample):
    ops, H_mat, U_mat, fixed_x, krylov_n = sample
    fixed_x = np.asarray(fixed_x, dtype=np.float64)
    E_ritz, psi0, phi = ritz_min_and_vector(num_qubit, ops, fixed_x, H_mat, U_mat, krylov_n, quantum_device)
    return float(E_ritz), psi0, phi


def get_sequence_energies_kqd(samples, num_workers=8):
    with Pool(processes=num_workers, initializer=_worker_init) as pool:
        ritz_results = pool.map(compute_ritz_energy_for_sample, samples)

    energies = np.array([res[0] for res in ritz_results], dtype=np.float32).reshape(-1, 1)
    return energies, ritz_results


def prepare_family_dataset(family_raw, num_qubit, t_evol, max_coefficient_value):
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


def build_prefix_batch(selected_family, sep_token_id, seq_per_hamiltonian, compute_device):
    prefix_list = []
    owner_indices = []

    for ham_idx, ham in enumerate(selected_family):
        prefix = np.concatenate([ham["coeff_tokens"], np.array([sep_token_id], dtype=np.int64)])
        for _ in range(seq_per_hamiltonian):
            prefix_list.append(prefix)
            owner_indices.append(ham_idx)

    prefix_batch = torch.tensor(np.stack(prefix_list), dtype=torch.long, device=compute_device)
    owner_indices = np.array(owner_indices, dtype=np.int64)
    return prefix_batch, owner_indices


if __name__ == "__main__":
    logger = setup_logger()
    compute_device = setup_gpu(COMPUTE_DEVICE, GPU_id)
    outdir = "results"
    os.makedirs(outdir, exist_ok=True)

    # ===== Hyperparameters(workers, GQE, KQD, Training, KQD etc) =====
    num_workers = 1  # 32

    num_feat = 8
    num_scale = 5
    gate_type = ["RX", "RY", "RZ", "CNOT", "H", "I", "MultiRZ"]
    param_scale = np.linspace(start=0.1, stop=1.0, num=num_scale)
    op_pool = make_op_pool(gate_type=gate_type, num_qubit=num_qubit, num_param=num_feat, param_scale=param_scale)
    op_pool_size = len(op_pool)

    max_coefficient_value = 5  # 10
    coeff_vocab_size = 2 * max_coefficient_value + 1
    sep_token_id = coeff_vocab_size
    gate_token_offset = coeff_vocab_size + 1
    vocab_size = gate_token_offset + op_pool_size

    coeff_names, backbone_ops = build_2local_2d_1local_backbone(num_qubit)
    coeff_len = len(coeff_names)
    prefix_len = coeff_len + 1

    train_num_hamiltonians = 10  # 500
    test_num_hamiltonians = 2  # 50
    train_seed = 1234
    test_seed = 5678

    train_hamiltonians_per_epoch = 8  # 32
    seq_per_hamiltonian = 4  # 32

    n_batches = min(train_hamiltonians_per_epoch, seq_per_hamiltonian)  # 32
    max_gate = 15  # 28?
    max_epoch = 6  # 2000
    T_max = 50  # 50
    T_min = 0.04

    krylov_n = 3
    t_evol = 0.2

    fixed_x = torch.linspace(0.1, 1.0, steps=num_feat).float()

    block_size = prefix_len + max_gate

    # ===== Preparing Hamiltonian =====
    train_family_raw = generate_hamiltonian_family(backbone_ops=backbone_ops, num_of_generation=train_num_hamiltonians,
                                                   seed=train_seed, max_coefficient_value=max_coefficient_value, )

    test_family_raw = generate_hamiltonian_family(backbone_ops=backbone_ops, num_of_generation=test_num_hamiltonians,
                                                  seed=test_seed, max_coefficient_value=max_coefficient_value, )

    train_family = prepare_family_dataset(train_family_raw, num_qubit=num_qubit, t_evol=t_evol,
                                          max_coefficient_value=max_coefficient_value, )

    test_family = prepare_family_dataset(test_family_raw, num_qubit=num_qubit, t_evol=t_evol,
                                         max_coefficient_value=max_coefficient_value, )

    # ===== Loading GQE model =====
    gpt = GPTQE(GPTConfig(vocab_size=vocab_size, block_size=block_size, dropout=0.2, bias=False)).to(compute_device)
    opt = gpt.configure_optimizers(weight_decay=0.01, lr=5e-5, betas=(0.9, 0.999), compute_device=compute_device)
    gpt.train()

    loss_history = []
    gap_history = []
    overlap0_history = []
    overlapk_history = []

    # ===== Training =====
    for i in range(max_epoch):
        logger.info(f"Starting epoch {i + 1}/{max_epoch}...")

        chosen_idx = np.random.choice(len(train_family), size=min(train_hamiltonians_per_epoch, len(train_family)),
                                      replace=False)
        selected_family = [train_family[idx] for idx in chosen_idx]

        prefix_batch, owner_indices = build_prefix_batch(
            selected_family=selected_family,
            sep_token_id=sep_token_id,
            seq_per_hamiltonian=seq_per_hamiltonian,
            compute_device=compute_device,
        )

        gpt.eval()
        train_token_seq_torch = gpt.generate_from_prefix(
            prefix_tokens=prefix_batch,
            max_new_tokens=max_gate,
            temperature=temperature(T_max=T_max, T_min=T_min, max_epoch=max_epoch, epoch=i),
            gate_token_offset=gate_token_offset,
        )
        gpt.train()

        train_token_seq = train_token_seq_torch.detach().cpu().numpy()
        train_gate_token_seq = train_token_seq[:, prefix_len:]
        train_op_seq = op_pool[train_gate_token_seq - gate_token_offset]

        samples = []
        for sample_idx, ops in enumerate(train_op_seq):
            ham = selected_family[owner_indices[sample_idx]]
            samples.append((ops, ham["H_mat"], ham["U_mat"], fixed_x, krylov_n))

        train_seq_en, train_ritz_results = get_sequence_energies_kqd(samples, num_workers=num_workers)

        tokens = torch.from_numpy(train_token_seq).to(compute_device)
        energies = torch.from_numpy(train_seq_en).to(compute_device)

        perm = torch.randperm(tokens.size(0), device=compute_device)
        tokens, energies = tokens[perm], energies[perm]

        token_batches = torch.tensor_split(tokens, n_batches)
        energy_batches = torch.tensor_split(energies, n_batches)

        loss_record = 0.0
        for token_batch, energy_batch in zip(token_batches, energy_batches):
            opt.zero_grad(set_to_none=True)
            loss = gpt.calculate_loss(token_batch, energy_batch, prefix_len=prefix_len)
            loss.backward()
            opt.step()
            loss_record += loss.item() / n_batches

        loss_history.append(loss_record)

        epoch_gap = []
        epoch_overlap0 = []
        epoch_overlapk = []

        for sample_idx, (E_ritz, psi0, phi) in enumerate(train_ritz_results):
            ham = selected_family[owner_indices[sample_idx]]

            epoch_gap.append(float(calc_energy_gap(E_ritz, ham["E0"])))
            epoch_overlap0.append(float(calc_overlap(psi0, ham["target_state"])))
            epoch_overlapk.append(float(calc_overlap(phi, ham["target_state"])))

        gap_history.append(float(np.mean(epoch_gap)))
        overlap0_history.append(float(np.mean(epoch_overlap0)))
        overlapk_history.append(float(np.mean(epoch_overlapk)))

        logger.info(f"Iter {i + 1} | loss={loss_record:.6f} | gap(mean)={gap_history[-1]:.6f} | "
                    f"overlap0(mean)={overlap0_history[-1]:.6f} | overlapK(mean)={overlapk_history[-1]:.6f}")

    logger.info("Training Done, Evaluation starting...")

    # ===== Save epochwise plots =====
    save_plt(loss_history, outpath=f"{outdir}/loss.png", title="loss", ylabel="loss")
    save_plt(gap_history, outpath=f"{outdir}/gap_mean.png", title="gap(mean)", ylabel="gap(mean)")
    save_plt(overlap0_history, outpath=f"{outdir}/overlap0_mean.png", title="overlap0(mean)", ylabel="overlap0(mean)")
    save_plt(overlapk_history, outpath=f"{outdir}/overlapk_mean.png", title="overlapK(mean)", ylabel="overlapK(mean)")

    # ===== Evaluation =====
    ns = list(range(krylov_n, krylov_n * 4))
    overlap_traces = []
    relative_gap_traces = []

    final_overlap_list = []
    final_relative_gap_list = []

    gpt.eval()

    for ham_idx, ham in enumerate(test_family, start=1):
        prefix = np.concatenate([ham["coeff_tokens"], np.array([sep_token_id], dtype=np.int64)])
        prefix_batch = torch.tensor(prefix[None, :], dtype=torch.long, device=compute_device)

        with torch.no_grad():
            test_token_seq_torch = gpt.generate_from_prefix(
                prefix_tokens=prefix_batch,
                max_new_tokens=max_gate,
                temperature=T_min,
                gate_token_offset=gate_token_offset,
            )

        test_token_seq = test_token_seq_torch.detach().cpu().numpy()
        ops = op_pool[test_token_seq[:, prefix_len:] - gate_token_offset][0]

        overlap_values = []
        relative_gap_values = []

        for n in ns:
            E_ritz, psi0, phi = ritz_min_and_vector(num_qubit, ops, fixed_x, ham["H_mat"], ham["U_mat"], n=n)
            overlap_values.append(float(calc_overlap(phi, ham["target_state"])))
            relative_gap_values.append(float(calc_relative_gap(E_ritz, ham["E0"])))

        overlap_traces.append({"x": ns, "y": overlap_values, "label": f"test_ham_{ham_idx}"})
        relative_gap_traces.append({"x": ns, "y": relative_gap_values, "label": f"test_ham_{ham_idx}"})

        final_overlap_list.append(float(overlap_values[-1]))
        final_relative_gap_list.append(float(relative_gap_values[-1]))

        logger.info(f"[test {ham_idx}/{len(test_family)}] "
                    f"overlapK(n={ns[-1]})={overlap_values[-1]:.6f} | r_gap(n={ns[-1]})={relative_gap_values[-1]:.6f}")

    save_plotly(traces=overlap_traces, title="overlap", xaxis_title="K-dim", yaxis_title="overlap",
                save_path=f"{outdir}/n_sweep_overlapK.html" )
    save_plotly(traces=relative_gap_traces, title="relative gap", xaxis_title="K-dim", yaxis_title="relative gap",
                save_path=f"{outdir}/n_sweep_relative_gap.html" )

    logger.info("Finished.")