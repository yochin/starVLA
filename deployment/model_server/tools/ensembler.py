# Copyright 2025 starVLA community. All rights reserved.
# Licensed under the MIT License.
"""Reusable Action Ensembler for Real-Robot Deployment & Simulation Evaluation.

Implements Temporal Ensembling and Adaptive Ensembling for action chunking
(e.g., ACT, Diffusion Policy, Flow Matching).

Supported modes:
  - "exp": Exponential decay weighting w_i = exp(-weight_decay * i) [ACT / standard]
  - "cosine": Cosine-similarity adaptive weighting [AdaptiveEnsembler]
  - "linear": Linear decay weighting
  - "uniform": Uniform average across all active chunks

Can be used in:
  1. Real-robot online control loop:
       ensembler = TemporalEnsembler(action_horizon=16, weight_decay=0.05)
       # at inference step:
       if step % infer_stride == 0:
           chunk = policy.predict_action(...) # (16, D)
           ensembler.add_chunk(chunk)
       # at every control step:
       action = ensembler.step() # (D,) -> send to robot

  2. Offline evaluation / trajectory generation:
       trajectory = ensembler.ensemble_trajectory(chunks_dict, start_step, end_step)
"""

from __future__ import annotations

from collections import deque
from typing import Dict, List, Optional, Sequence, Union

import numpy as np


class TemporalEnsembler:
    """General Temporal Action Ensembler for streaming or offline execution."""

    def __init__(
        self,
        action_horizon: int = 16,
        mode: str = "exp",
        weight_decay: float = 0.05,
        adaptive_alpha: float = 0.1,
        discrete_dims: Optional[Sequence[int]] = None,
    ) -> None:
        """
        Args:
            action_horizon: Length of each predicted action chunk (e.g. 16).
            mode: Ensembling mode: 'exp', 'cosine', 'linear', or 'uniform'.
            weight_decay: Decay rate for 'exp' mode (0.0 = uniform).
            adaptive_alpha: Scaling factor for 'cosine' similarity mode.
            discrete_dims: Dimension indices to round to nearest int (e.g. [10, 36] for hand modes).
        """
        self.action_horizon = int(action_horizon)
        self.mode = mode.lower()
        self.weight_decay = float(weight_decay)
        self.adaptive_alpha = float(adaptive_alpha)
        self.discrete_dims = list(discrete_dims) if discrete_dims is not None else []

        self.history: List[dict] = []
        self.current_step = 0

    def reset(self, start_step: int = 0) -> None:
        """Clear action history and reset step counter."""
        self.history.clear()
        self.current_step = int(start_step)

    def add_chunk(self, chunk: np.ndarray, start_step: Optional[int] = None) -> None:
        """Add a new action chunk to the ensembler history.

        Args:
            chunk: Array of shape (H, D) or (1, H, D).
            start_step: Timestep at which this chunk begins. Defaults to self.current_step.
        """
        chunk_arr = np.asarray(chunk, dtype=np.float32)
        if chunk_arr.ndim == 3 and chunk_arr.shape[0] == 1:
            chunk_arr = chunk_arr[0]
        assert chunk_arr.ndim == 2, f"Expected chunk shape (H, D); got {chunk_arr.shape}"

        step_idx = self.current_step if start_step is None else int(start_step)
        self.history.append({
            "start_step": step_idx,
            "actions": chunk_arr,
            "length": len(chunk_arr),
        })

    def step(self) -> np.ndarray:
        """Compute the ensembled action for current_step and advance current_step by 1."""
        action = self.get_action(self.current_step)
        self.current_step += 1
        return action

    def get_action(self, target_step: int) -> np.ndarray:
        """Compute the blended action for a specific target timestep."""
        # Prune chunks that no longer overlap with target_step
        self.history = [h for h in self.history if h["start_step"] + h["length"] > target_step]

        candidates = []
        ages = []  # age of prediction (how many steps ago this chunk was generated)

        for item in self.history:
            offset = target_step - item["start_step"]
            if 0 <= offset < item["length"]:
                candidates.append(item["actions"][offset])
                ages.append(offset)

        if not candidates:
            raise ValueError(
                f"TemporalEnsembler: No active prediction chunks available for step {target_step}."
            )

        preds = np.stack(candidates, axis=0)  # (N, D)
        N, D = preds.shape

        if N == 1:
            blended = preds[0].copy()
        else:
            ages = np.array(ages, dtype=np.float32)
            if self.mode == "exp":
                # Newer predictions (smaller age / offset) get higher weight
                weights = np.exp(-self.weight_decay * ages)
            elif self.mode == "linear":
                weights = np.maximum(0.01, 1.0 - ages / self.action_horizon)
            elif self.mode == "cosine":
                ref = preds[-1]  # most recent chunk prediction
                dot = np.sum(preds * ref, axis=1)
                norm_p = np.linalg.norm(preds, axis=1)
                norm_r = np.linalg.norm(ref)
                cos_sim = dot / (norm_p * norm_r + 1e-7)
                weights = np.exp(self.adaptive_alpha * cos_sim)
            elif self.mode == "uniform":
                weights = np.ones(N, dtype=np.float32)
            else:
                raise ValueError(f"Unknown ensembling mode: {self.mode}")

            weights = weights / np.sum(weights)
            blended = np.sum(weights[:, None] * preds, axis=0)

        # Handle discrete channels (e.g. hand modes)
        for d in self.discrete_dims:
            if d < D:
                blended[d] = float(np.round(blended[d]))

        return blended

    def ensemble_trajectory(
        self,
        chunks: Dict[int, np.ndarray],
        start_step: int,
        num_steps: int,
    ) -> np.ndarray:
        """Helper to ensemble a complete sequence of predictions into a full trajectory.

        Args:
            chunks: Dict mapping start_step (int) -> action chunk np.ndarray (H, D).
            start_step: First timestep of the output trajectory.
            num_steps: Total number of timesteps to generate.

        Returns:
            np.ndarray of shape (num_steps, D).
        """
        self.reset(start_step=start_step)
        sorted_starts = sorted(chunks.keys())
        out_actions = []

        chunk_ptr = 0
        for step in range(start_step, start_step + num_steps):
            # Push all chunks that start at or before current step
            while chunk_ptr < len(sorted_starts) and sorted_starts[chunk_ptr] <= step:
                s = sorted_starts[chunk_ptr]
                self.add_chunk(chunks[s], start_step=s)
                chunk_ptr += 1

            act = self.step()
            out_actions.append(act)

        return np.stack(out_actions, axis=0)


class AdaptiveEnsembler:
    """Drop-in compatibility adapter for existing AdaptiveEnsembler usages."""

    def __init__(self, pred_action_horizon: int, adaptive_ensemble_alpha: float = 0.1) -> None:
        self._inner = TemporalEnsembler(
            action_horizon=pred_action_horizon,
            mode="cosine",
            adaptive_alpha=adaptive_ensemble_alpha,
            discrete_dims=[10, 36],
        )

    def reset(self) -> None:
        self._inner.reset()

    def ensemble_action(self, cur_action: np.ndarray) -> np.ndarray:
        self._inner.add_chunk(cur_action)
        return self._inner.step()

    def step(self) -> np.ndarray:
        return self._inner.step()
