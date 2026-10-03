# Copyright 2025 starVLA community. All rights reserved.
# Licensed under the MIT License, Version 1.0 (the "License");
# Implemented by [Jinhui YE / HKUST University] in [2025].

"""Implementations of various action heads, which serve as alternatives to VLM sequential token prediction."""

"this file is adap from https://github.com/moojink/openvla-oft/blob/main/prismatic/models/action_heads.py"

import torch
import torch.nn as nn

from starVLA.training.trainer_utils import initialize_overwatch

logger = initialize_overwatch(__name__)


class MLPResNetBlock(nn.Module):
    """One MLP ResNet block with a residual connection."""

    def __init__(self, dim):
        super().__init__()
        self.dim = dim
        self.ffn = nn.Sequential(  # feedforward network, similar to the ones in Transformers
            nn.LayerNorm(dim),
            nn.Linear(dim, dim),
            nn.ReLU(),
        )

    def forward(self, x):
        # x: (batch_size, hidden_dim)
        # We follow the module ordering of "Pre-Layer Normalization" feedforward networks in Transformers as
        # described here: https://arxiv.org/pdf/2002.04745.pdf
        identity = x
        x = self.ffn(x)
        x = x + identity
        return x


class MLPResNet(nn.Module):
    """MLP with residual connection blocks."""

    def __init__(self, num_blocks, input_dim, hidden_dim, output_dim):
        super().__init__()
        self.layer_norm1 = nn.LayerNorm(input_dim)
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.relu = nn.ReLU()
        self.mlp_resnet_blocks = nn.ModuleList()
        for _ in range(num_blocks):
            self.mlp_resnet_blocks.append(MLPResNetBlock(dim=hidden_dim))
        self.layer_norm2 = nn.LayerNorm(hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, output_dim)

    def forward(self, x):
        # x: (batch_size, input_dim)
        x = self.layer_norm1(x)  # shape: (batch_size, input_dim)
        x = self.fc1(x)  # shape: (batch_size, hidden_dim)
        x = self.relu(x)  # shape: (batch_size, hidden_dim)
        for block in self.mlp_resnet_blocks:
            x = block(x)  # shape: (batch_size, hidden_dim)
        x = self.layer_norm2(x)  # shape: (batch_size, hidden_dim)
        x = self.fc2(x)  # shape: (batch_size, output_dim)
        return x


class L1RegressionActionHead(nn.Module):
    """Simple MLP-based action head that generates continuous actions via L1 regression.

    ``logit_dim > 0`` splits the output across two independent MLPResNets: the first
    emits the leading ``action_dim - logit_dim`` regression dims, the second the
    trailing ``logit_dim`` classification logits. The concatenation keeps the same
    layout, so nothing downstream changes.

    Why the split exists. With one shared network the two output groups demand
    magnitudes two to three orders of magnitude apart: measured on the G1 hand run,
    the body deltas sit at 0.003-0.012 in normalised units while the hand logits span
    [-29.9, +14.7] with standard deviation 11. The shared fc2 showed it -- logit rows
    had norm 4.35 against 0.53 for the body rows, 8.1x -- and the body's own rows were
    *not* shrunk (0.95-1.01x of the run without a classifier), so the same weights were
    producing smaller outputs. That points at the shared hidden features being pulled
    toward the logits' high-variance directions. In that run every body part collapsed
    to roughly the persistence baseline while a run without the classifier reached
    0.90-0.94x of it.

    Default ``logit_dim=0`` keeps a single network and the original parameter names, so
    existing checkpoints load unchanged.
    """

    def __init__(
        self,
        input_dim=2048,
        hidden_dim=4096,
        action_dim=7,
        NUM_ACTIONS_CHUNK=8,
        logit_dim=0,
    ):
        super().__init__()
        self.action_dim = action_dim
        self.NUM_ACTIONS_CHUNK = NUM_ACTIONS_CHUNK
        self.logit_dim = int(logit_dim)
        if self.logit_dim < 0 or self.logit_dim >= action_dim:
            raise ValueError(
                f"logit_dim must be in [0, action_dim); got {logit_dim} with "
                f"action_dim={action_dim}."
            )
        self.body_dim = action_dim - self.logit_dim

        self.model = MLPResNet(
            num_blocks=2, input_dim=input_dim, hidden_dim=hidden_dim, output_dim=self.body_dim
        )
        self.logit_model = (
            MLPResNet(
                num_blocks=2, input_dim=input_dim, hidden_dim=hidden_dim, output_dim=self.logit_dim
            )
            if self.logit_dim > 0
            else None
        )

    def predict_action(self, actions_hidden_states):
        """
        actions_hidden_states: (B, chunk_len, hidden_dim)
        Returns: (B, chunk_len, action_dim)
        """
        batch_size, chunk_len, hidden_dim = actions_hidden_states.shape
        x = actions_hidden_states.reshape(batch_size * chunk_len, hidden_dim)
        out = self.model(x)  # (B * chunk_len, body_dim)
        if self.logit_model is not None:
            out = torch.cat([out, self.logit_model(x)], dim=-1)
        actions = out.view(batch_size, chunk_len, self.action_dim)
        return actions

    def forward(self, actions_hidden_states):
        return self.predict_action(actions_hidden_states)


def get_action_model(config=None):
    """
    Factory: build ActionModel from global framework config.

    Args:
        config: Global config (expects config.framework.action_model namespace).
    Returns:
        ActionModel: an initialised L1 regression head. Deterministic -- it maps the
            VLM hidden states straight to actions, with no noise and no sampling step.
    """
    action_model_cfg = config.framework.action_model
    model_type = action_model_cfg.action_model_type
    action_hidden_dim = action_model_cfg.action_hidden_dim
    action_dim = action_model_cfg.action_dim
    # `action_horizon` is the canonical chunk length, normalised upstream
    # by share_tools.apply_config_compat.
    action_horizon = int(action_model_cfg.action_horizon)

    # Opt-in: give the discrete hand logits their own MLPResNet instead of sharing the
    # body's. Off by default, so every existing run and checkpoint is unaffected.
    # The split must match the framework's, which takes the leading
    # action_dim - hand_count * 6 dims as the body.
    def _get(key, default):
        return (
            action_model_cfg.get(key, default) if hasattr(action_model_cfg, "get")
            else getattr(action_model_cfg, key, default)
        )

    logit_dim = 0
    if bool(_get("hand_head", False)) and bool(_get("hand_head_separate", False)):
        logit_dim = int(_get("hand_count", 2)) * 6  # on(1) + mode(4) + thumb_flex(1)
        logger.info(
            "MLP head: hand logits get their own MLPResNet — body %d dims, logits %d dims.",
            action_dim - logit_dim, logit_dim,
        )

    action_model = L1RegressionActionHead(
        input_dim=action_hidden_dim,
        hidden_dim=action_hidden_dim * 2,
        action_dim=action_dim,
        NUM_ACTIONS_CHUNK=action_horizon,
        logit_dim=logit_dim,
    )

    return action_model
