"""One PLR + MLP adapter using the authors' numerical embedding package.

Torch and rtdl are imported only when this adapter is constructed. Fixed epochs
use the last weights: no validation, scheduler, calibration or checkpoint search.
"""

from __future__ import annotations

import os
from pathlib import Path

import numpy as np


class NumericalEmbeddingMLP:
    def __init__(self, parameters, *, device):
        import torch
        import rtdl_num_embeddings

        self.torch = torch
        self.embeddings = rtdl_num_embeddings
        self.parameters = dict(parameters)
        self.device = "cuda" if device in {"gpu", "cuda"} else "cpu"
        self.classes_ = np.array([0, 1])
        self.model = None

    def _make_network(self):
        torch = self.torch
        nn = torch.nn
        p = self.parameters
        # Directly use the canonical non-lite PLR: feature-specific frequencies,
        # cosine/sine, feature-specific linear projection, then ReLU.
        embedding = self.embeddings.PeriodicEmbeddings(
            59, d_embedding=p["d_embedding"], n_frequencies=p["n_frequencies"],
            frequency_init_scale=p["frequency_init_scale"], activation=True, lite=False,
        )
        layers = [embedding, nn.Flatten(start_dim=1)]
        dimension = 59 * p["d_embedding"]
        # Match the authors' rtdl-revisiting-models MLP block structure; widths
        # and dropout are the frozen screen recipe recorded in config metadata.
        for width in p["hidden_sizes"]:
            layers.extend([nn.Linear(dimension, width), nn.ReLU(), nn.Dropout(p["dropout"])])
            dimension = width
        layers.append(nn.Linear(dimension, 1))
        return nn.Sequential(*layers).to(device=self.device, dtype=torch.float32)

    def fit(self, X_train, y_train):
        torch = self.torch
        p = self.parameters
        if X_train.dtype != np.float32 or X_train.shape != (len(y_train), 59):
            raise ValueError("PLR expects standardized float32 59D training rows")
        if self.device == "cuda":
            os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
            torch.cuda.synchronize()
        deterministic = torch.are_deterministic_algorithms_enabled()
        warn_only = torch.is_deterministic_algorithms_warn_only_enabled()
        cudnn_benchmark = torch.backends.cudnn.benchmark
        cudnn_deterministic = torch.backends.cudnn.deterministic
        default_dtype = torch.get_default_dtype()
        devices = [torch.cuda.current_device()] if self.device == "cuda" else []
        try:
            # Restore caller RNG/determinism state afterwards; independent
            # candidate runs must not change another candidate's random stream.
            with torch.random.fork_rng(devices=devices):
                torch.manual_seed(p["random_state"])
                torch.set_default_dtype(torch.float32)
                torch.use_deterministic_algorithms(p["deterministic"])
                torch.backends.cudnn.benchmark = False
                torch.backends.cudnn.deterministic = True
                self.model = self._make_network()
                optimizer = torch.optim.AdamW(
                    self.model.parameters(), lr=p["learning_rate"], weight_decay=p["weight_decay"],
                    betas=tuple(p["adam_betas"]), eps=p["adam_eps"],
                )
                criterion = torch.nn.BCEWithLogitsLoss()
                random = np.random.default_rng(p["random_state"])
                self.loss_curve_ = []
                self.model.train()
                for _ in range(p["max_epochs"]):
                    indices = random.permutation(len(y_train))
                    total_loss = torch.zeros((), device=self.device)
                    for start in range(0, len(indices), p["batch_size"]):
                        batch = indices[start:start + p["batch_size"]]
                        X = torch.from_numpy(X_train[batch]).to(self.device)
                        y = torch.as_tensor(y_train[batch], dtype=torch.float32, device=self.device)
                        optimizer.zero_grad(set_to_none=True)
                        loss = criterion(self.model(X).squeeze(-1), y)
                        loss.backward()
                        optimizer.step()
                        total_loss += loss.detach() * len(batch)
                    self.loss_curve_.append(float(total_loss.cpu()) / len(y_train))
                self.n_iter_ = p["max_epochs"]
                self.loss_ = self.loss_curve_[-1]
                del optimizer
                if self.device == "cuda":
                    torch.cuda.synchronize()
        finally:
            torch.use_deterministic_algorithms(deterministic, warn_only=warn_only)
            torch.backends.cudnn.benchmark = cudnn_benchmark
            torch.backends.cudnn.deterministic = cudnn_deterministic
            torch.set_default_dtype(default_dtype)
        return self

    def predict_proba(self, X):
        torch = self.torch
        self.model.eval()
        p1 = np.empty(len(X), dtype=np.float64)
        with torch.inference_mode():
            for start in range(0, len(X), self.parameters["batch_size"]):
                end = min(start + self.parameters["batch_size"], len(X))
                batch = torch.from_numpy(X[start:end]).to(self.device)
                p1[start:end] = torch.sigmoid(self.model(batch).squeeze(-1)).cpu().numpy()
        return np.column_stack([1 - p1, p1])

    def save(self, path):
        from .mission2_clean_stage4 import clean_59d_feature_names

        path = Path(path)
        with path.open("xb") as stream:
            self.torch.save({"state_dict": {k: v.detach().cpu() for k, v in self.model.state_dict().items()},
                             "parameters": self.parameters, "feature_names": clean_59d_feature_names(),
                             "input": "train-only standardized 59D float32",
                             "loss_curve": self.loss_curve_}, stream)

    def release(self):
        self.model = None
        if self.device == "cuda":
            # Frees unused allocations only; live tensors belonging to other
            # objects are retained by Torch's allocator.
            self.torch.cuda.empty_cache()
