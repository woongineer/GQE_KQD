import csv
import logging
import os
import sys

import matplotlib.pyplot as plt
import plotly.graph_objects as go
import torch


def setup_gpu(COMPUTE_DEVICE, GPU_id):
    if COMPUTE_DEVICE == "cuda":
        os.environ["CUDA_VISIBLE_DEVICES"] = str(GPU_id)
        torch.cuda.set_device(GPU_id)
        torch.backends.cudnn.benchmark = True
        torch.set_float32_matmul_precision("high")
        return torch.device("cuda")
    return torch.device("cpu")


def setup_logger():
    logger = logging.getLogger()
    logger.setLevel(logging.INFO)

    if not logger.handlers:
        handler = logging.StreamHandler(sys.stdout)
        formatter = logging.Formatter("%(asctime)s - %(levelname)s - %(message)s")
        handler.setFormatter(formatter)
        logger.addHandler(handler)

    return logger


def save_plt(y, outpath, title, xlabel='epoch', ylabel=None):
    plt.figure(figsize=(10, 5))
    plt.plot(y)
    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(outpath)
    plt.close()


def save_plotly(traces, title, xaxis_title, yaxis_title, save_path):
    fig = go.Figure()

    for tr in traces:
        fig.add_trace(
            go.Scatter(
                x=tr["x"],
                y=tr["y"],
                mode="lines+markers",
                name=tr["label"],
            )
        )

    fig.update_layout(
        title=title,
        xaxis_title=xaxis_title,
        yaxis_title=yaxis_title,
        hovermode="x unified",
        template="plotly_white",
    )

    fig.write_html(save_path)


def save_csv(nsweep_results, save_path):
    with open(save_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["metric", "rank", "epoch", "label", "n", "value"])

        for metric_name, traces in nsweep_results.items():
            for tr in traces:
                rank = tr["rank"]
                epoch = tr["epoch"]
                label = tr["label"]

                for n_val, y_val in zip(tr["x"], tr["y"]):
                    writer.writerow([metric_name, rank, epoch, label, n_val, y_val])
