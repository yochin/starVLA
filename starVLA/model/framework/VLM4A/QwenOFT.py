# Copyright 2025 starVLA community. All rights reserved.
# Licensed under the MIT License, Version 1.0 (the "License");
# Implemented by [Jinhui YE / HKUST University] in [2025].

"""
Qwen-OFT Framework

A lightweight implementation that uses an action special token to parallelly predict continuous actions
conditioned on multi-view images plus a language instruction (shares parameters with the VLM).
Inspired by OpenVLA-OFT
Key Points:
  - Qwen2.5 vision-language backbone
  - Injects an action special token into the VLM
  - Continuous action prediction via L1 regression over the action special token hidden states


Note: How to add special tokens to Qwen2.5:
  download our model checkpoint with special tokens added: https://huggingface.co/StarVLA/Qwen2.5-VL-3B-Instruct-Action
  or /starVLA/model/modules/vlm/tools/add_qwen_special_tokens/README.md (adapt a little code)

"""

from dataclasses import dataclass, field
from typing import List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
from PIL import Image

from deployment.model_server.tools.image_tools import prepare_multiview_images, to_pil_preserve
from starVLA.model.tools import FRAMEWORK_REGISTRY
from starVLA.training.trainer_utils import initialize_overwatch

logger = initialize_overwatch(__name__)

# HuggingFace Default / LLaMa-2 IGNORE_INDEX (for labels)
IGNORE_INDEX = -100

from starVLA.model.framework.base_framework import baseframework
from starVLA.model.framework.share_tools import add_discretized_state_to_instruction, merge_framework_config
from starVLA.model.modules.action_model.MLP_ActionHeader import get_action_model
from starVLA.model.modules.vlm import get_vlm_model


# ──────────────────────────────────────────────────────────────────────
#  Default Config for QwenOFT
#  - Documents every framework-level parameter with type + description
#  - YAML values override these defaults; extra YAML keys are preserved
# ──────────────────────────────────────────────────────────────────────
@dataclass
class QwenOFTDefaultConfig:
    """QwenOFT framework default parameters.

    All fields can be overridden by the corresponding key in the YAML
    ``framework:`` section.  Extra YAML keys not listed here are kept
    as-is (Config-as-API flexibility).
    """

    # --- Registry identifier (must match @FRAMEWORK_REGISTRY.register) ---
    name: str = "QwenOFT"

    # === VLM backbone (Qwen2.5-VL / Qwen3-VL) ===
    qwenvl: dict = field(
        default_factory=lambda: {
            # Path to base VLM checkpoint (local or HF hub id)
            "base_vlm": "./playground/Pretrained_models/Qwen3-VL-4B-Instruct-Action",
            # Attention implementation: "flash_attention_2" | "eager" | "sdpa"
            "attn_implementation": "flash_attention_2",
        }
    )

    # === Action head (MLP regression over action special tokens) ===
    action_model: dict = field(
        default_factory=lambda: {
            # Action head architecture type
            "action_model_type": "MLP",
            # Dimensionality of each action vector (e.g., 7 for 6-DoF + gripper)
            "action_dim": 7,
            # Hidden dim for the action MLP (auto-set from VLM hidden_size at runtime)
            "action_hidden_dim": 2560,
            # How many future steps to predict
            "future_action_window_size": 8,
            # How many past steps included in action chunk (usually 0)
            "past_action_window_size": 0,
        }
    )


@FRAMEWORK_REGISTRY.register("QwenOFT")
class Qwenvl_OFT(baseframework):
    """
    Multimodal vision-language-action model (OFT variant).

    Components:
      - Qwen2.5-VL / Qwen3-VL backbone for fused language/vision token embeddings
      - Action special token injected into the VLM sequence
      - MLP regression head over action token hidden states (L1 loss)

    Focus: Predict future continuous actions conditioned on images + instruction.
    """

    def __init__(
        self,
        config: Optional[dict] = None,
        **kwargs,
    ) -> None:
        """
        Construct all submodules and cache key configuration values.

        Args:
            config: Hierarchical configuration (OmegaConf/dict) containing framework + trainer sections.
            **kwargs: Reserved for future overrides (unused).
        """
        super().__init__()
        # Merge framework defaults with YAML config (YAML wins on conflicts)
        self.config = merge_framework_config(QwenOFTDefaultConfig, config)
        self.qwen_vl_interface = get_vlm_model(config=self.config)
        # align action_hidden_dim to VLM hidden_size at runtime
        self.config.framework.action_model.action_hidden_dim = self.qwen_vl_interface.model.config.hidden_size
        self.action_model = get_action_model(config=self.config)

        # `action_horizon` is the single source of truth for chunk length.
        # Legacy aliases (`future_action_window_size`, `past_action_window_size`)
        # are normalised upstream by `share_tools.apply_config_compat`, so we
        # only ever read `action_horizon` here.
        self.action_horizon = int(self.config.framework.action_model.action_horizon)
        self.chunk_len = self.action_horizon
        # self.hidden_dim = config.framework.action_model.action_hidden_dim

        self.action_token = "🔍"  # TODO also can add spacail token to Qwen, but too complex
        self.action_token_id = self.qwen_vl_interface.processor.tokenizer("🔍", add_special_tokens=False)["input_ids"][0]
        # L1 loss
        self.l1_loss = nn.L1Loss()

        # Action loss mask support for ignored action dims
        act_cfg = self.config.framework.action_model
        ignored_dims = act_cfg.get("ignored_action_dims", None) if hasattr(act_cfg, "get") else getattr(act_cfg, "ignored_action_dims", None)
        action_dim = int(act_cfg.action_dim)
        if ignored_dims is not None and len(ignored_dims) > 0:
            mask = torch.ones(action_dim, dtype=torch.float32)
            mask[list(ignored_dims)] = 0.0
            self.register_buffer("action_loss_mask", mask)
            logger.info(f"QwenOFT: Ignoring action dimensions {list(ignored_dims)} in L1 loss. Active dims: {int(mask.sum().item())}/{action_dim}")
        else:
            self.action_loss_mask = None

        # Optional quaternion handling, disabled unless quaternion_dims is set.
        # L1 on raw quaternion components respects neither the unit-norm
        # constraint nor the double cover (q and -q are the same rotation), so
        # when enabled those dims are scored with 1 - |<q_pred, q_target>| and
        # dropped from the L1 term instead.
        quat_dims = act_cfg.get("quaternion_dims", None) if hasattr(act_cfg, "get") else getattr(act_cfg, "quaternion_dims", None)
        self.quaternion_slice = None
        self.quaternion_loss_weight = 1.0
        if quat_dims is not None and len(quat_dims) == 2:
            self.quaternion_slice = (int(quat_dims[0]), int(quat_dims[1]))
            self.quaternion_loss_weight = float(
                act_cfg.get("quaternion_loss_weight", 1.0) if hasattr(act_cfg, "get")
                else getattr(act_cfg, "quaternion_loss_weight", 1.0)
            )
            logger.info(
                f"QwenOFT: dims {self.quaternion_slice} scored as a quaternion "
                f"(weight {self.quaternion_loss_weight}), excluded from the L1 term."
            )

        # Optional residual parameterisation: the head predicts a delta which is
        # added to the state the model was given, instead of the absolute target.
        # Off unless residual_from_state is set, in which case nothing below the
        # head changes - the target stays absolute, so every eval and deployment
        # path is untouched.
        #
        # Why: measured on the 160k 6D run, the model's error barely grows with
        # the horizon (val RMSE 0.1056 at 0.07 s to 0.1377 at 0.57 s) while the
        # persistence baseline's triples (0.0444 to 0.1401). A flat error is the
        # signature of a fixed floor in placing the robot's absolute joint
        # configuration from pixels, not of failing to predict motion. Handing the
        # model state[t] lets it answer "here plus a small delta" instead.
        self.residual_from_state = bool(
            act_cfg.get("residual_from_state", False) if hasattr(act_cfg, "get")
            else getattr(act_cfg, "residual_from_state", False)
        )
        if self.residual_from_state:
            logger.info(
                "QwenOFT: predicting a residual from the input state "
                "(head output is added to state[t] before the loss)."
            )

        # Whether a discretised copy of the state is also prepended to the prompt
        # (π₀.5 style). Lives under framework.action_model rather than with the
        # other data flags so it lands in the saved config.yaml and inference
        # makes the same choice training did; read from datasets.vla_data it is
        # absent at inference and silently defaults back on. True keeps the prior
        # behaviour whenever a state is present.
        self.state_in_instruction = bool(
            act_cfg.get("state_in_instruction", True) if hasattr(act_cfg, "get")
            else getattr(act_cfg, "state_in_instruction", True)
        )
        if not self.state_in_instruction:
            logger.info(
                "QwenOFT: state is NOT prepended to the instruction; it reaches "
                "the model only through the residual connection."
            )

        # Optional discrete hand head. The hands are a rare-event detection problem,
        # not a regression one: 97% of 0.53 s chunks show no hand change and the rest
        # are near-full transitions, so L1 settles on the conditional median — "no
        # change" — and predicts nothing. Measured on the 160k run, the hands sat at
        # exactly 1.00x the persistence baseline while arms and legs reached 0.54x.
        #
        # So the hands get their own classification outputs, appended after the body
        # dims: per hand one on/off logit, four mode logits, one thumb-flexion logit.
        # The residual and the L1 term cover only the body dims; adding state[t] to a
        # logit would be meaningless.
        self.hand_head = bool(
            act_cfg.get("hand_head", False) if hasattr(act_cfg, "get")
            else getattr(act_cfg, "hand_head", False)
        )
        self.hand_n = 0
        self.hand_block = 6  # on(1) + mode(4) + thumb_flex(1)
        self.body_dim = action_dim
        self.hand_mode_n = 4
        if self.hand_head:
            self.hand_n = int(
                act_cfg.get("hand_count", 2) if hasattr(act_cfg, "get")
                else getattr(act_cfg, "hand_count", 2)
            )
            self.body_dim = action_dim - self.hand_n * self.hand_block
            if self.body_dim <= 0:
                raise ValueError(
                    f"hand_head=True needs action_dim ({action_dim}) to exceed the hand "
                    f"logits ({self.hand_n * self.hand_block}); body_dim came out "
                    f"{self.body_dim}."
                )
            # 전이는 스텝의 1.4~1.6% 뿐이라 가중 없이는 "변화 없음"만 예측한다.
            self.hand_on_pos_weight = float(
                act_cfg.get("hand_on_pos_weight", 1.0) if hasattr(act_cfg, "get")
                else getattr(act_cfg, "hand_on_pos_weight", 1.0)
            )
            self.hand_loss_weight = float(
                act_cfg.get("hand_loss_weight", 1.0) if hasattr(act_cfg, "get")
                else getattr(act_cfg, "hand_loss_weight", 1.0)
            )
            logger.info(
                "QwenOFT: discrete hand head on — body dims 0:%d, %d hand(s) x "
                "%d logits after that (on/mode4/thumb_flex). on pos_weight=%.2f, "
                "hand loss weight=%.2f",
                self.body_dim, self.hand_n, self.hand_block,
                self.hand_on_pos_weight, self.hand_loss_weight,
            )

    def forward(
        self,
        examples: List[dict] = None,
        **kwargs,
    ) -> Tuple:
        """
        Training forward: directly regress future actions (no diffusion).

        Flow:
          1. Build QwenVL inputs (images + instruction tokens)
          2. Extract hidden states from configured layer range
          7. Predict action and compute L1 loss

        Args:
            examples: List[dict], each dict requires:
                - image: List[PIL.Image] (multi-view)
                - lang: str instruction
                - action: np.ndarray or list shaped [T, action_dim]
            **kwargs: Reserved.

        Returns:
            dict:
                action_loss (torch.Tensor): Scalar diffusion noise prediction loss.
        """
        batch_images = [example["image"] for example in examples]  #  [B, [PLT]]
        instructions = [example["lang"] for example in examples]  # [B, str]
        actions = [example["action"] for example in examples]  # label [B, len, 7]
        data_cfg = self.config.datasets.vla_data
        # 손 헤드가 켜져 있으면 hand_prev 는 필수다. 없으면 프롬프트 접두사가 조용히
        # 빠져 학습과 다른 입력이 되고, 오류 없이 성능만 떨어진다.
        if self.hand_head and "hand_prev" not in examples[0]:
            raise ValueError(
                "hand_head=True requires 'hand_prev' in each example — the hand pose "
                "the robot is currently holding, as (on, mode, thumb_flex) per hand. "
                "Training always supplies it from action[t-1]; omitting it at inference "
                "silently drops the prompt suffix the model was trained with."
            )
        hand_prev = (
            [example["hand_prev"] for example in examples] if self.hand_head else None
        )

        # Residual mode needs the state regardless of what the saved config says.
        # config.yaml only records the keys the trainer accessed, and
        # include_state is not one of them, so gating on it here drops the state
        # at inference and the residual has nothing to add - a failure that only
        # shows up at eval time.
        use_state = bool(getattr(data_cfg, "include_state", False)) or self.residual_from_state
        state = (
            [example["state"] for example in examples]
            if use_state and "state" in examples[0]
            else None
        )  # List[ndarray (1, state_dim)] or None

        # Optionally prepend discretised proprioceptive state tokens to each instruction (π₀.5 style).
        # Defaults to on whenever a state is present, as before. Resolved in
        # __init__ from framework.action_model so the choice survives into the
        # saved config.yaml: read from datasets.vla_data it would be absent at
        # inference and silently default back on, adding a prompt prefix the
        # training run never had.
        if state is not None and self.state_in_instruction:
            instructions = self.add_discretized_state_to_instruction(instructions, state)

        # 이전 손 명령을 프롬프트에 붙인다. 관절에서 잔차가 "절대 자세를 픽셀에서
        # 추정"하는 부담을 없앤 것과 같은 역할로, 손은 "지금 무엇을 쥐고 있는가"를
        # 알려 주어 과제를 "유지할지 바꿀지"로 좁힌다. action[t-1] 을 쓰는 이유는
        # action[t] 가 첫 타깃 스텝이라 누출이기 때문이다.
        if hand_prev is not None:
            instructions = [
                instr + self._hand_prev_suffix(hp)
                for instr, hp in zip(instructions, hand_prev)
            ]

        # step 0: add special action token to instruction
        action_tokens = (
            self.action_token * self.chunk_len
        )  # can't add " " between two tokens, otherwise will be tokenized to multiple tokens
        prompt_suffix = f" Please predict the next {self.chunk_len} robot actions: <action>{action_tokens}<action>."
        instructions = [instruction + prompt_suffix for instruction in instructions]

        # Step 1: QWenVL input format
        qwen_inputs = self.qwen_vl_interface.build_qwenvl_inputs(images=batch_images, instructions=instructions)
        with torch.autocast("cuda", dtype=torch.bfloat16):
            qwenvl_outputs = self.qwen_vl_interface(
                **qwen_inputs,
                output_attentions=False,
                output_hidden_states=True,
                return_dict=True,
            )
            # last_hidden_state: [B, seq_len, H]
            last_hidden = qwenvl_outputs.hidden_states[-1]  # [B, L, H]

        # Step 4: Action Expert Forward and Loss
        with torch.autocast("cuda", dtype=torch.float32):
            # Extract action token embeddings as action prediction queries
            input_ids = qwen_inputs.get("input_ids", None)
            action_queries = self._gather_action_token_embeddings(
                last_hidden, input_ids, action_token_id=self.action_token_id
            )  # [B, chunk_len, H]
            pred_actions = self.action_model.predict_action(action_queries)  # (B, chunk_len, action_dim)
            pred_actions = self._apply_state_residual(pred_actions, state)

            # Label alignment: take the last chunk_len segment
            actions = torch.tensor(
                np.array(actions), device=pred_actions.device, dtype=pred_actions.dtype
            )  # [B, T_full, action_dim]
            actions_target = actions[:, -self.action_horizon :, :]  # (B, action_horizon, action_dim)

            # 손 헤드가 켜져 있으면 뒤쪽 로짓 구간을 떼어 낸다. L1 은 body 에만 적용한다.
            if self.hand_head:
                body_pred = pred_actions[..., : self.body_dim]
                hand_logits = pred_actions[..., self.body_dim :]
            else:
                body_pred, hand_logits = pred_actions, None

            # Compute L1 loss
            diff = torch.abs(body_pred - actions_target)
            if self.action_loss_mask is not None:
                loss_mask = self.action_loss_mask.to(device=diff.device, dtype=diff.dtype)[: diff.shape[-1]]
                action_loss = (diff * loss_mask).sum() / (loss_mask.sum() * diff.shape[0] * diff.shape[1])
            else:
                action_loss = diff.mean()
            per_dim_loss = diff.mean(dim=(0, 1)).detach()

            # Opt-in: re-score the quaternion dims with a rotation-aware term.
            # Skipped entirely when quaternion_dims is unset, leaving action_loss
            # exactly as computed above.
            if self.quaternion_slice is not None:
                qs, qe = self.quaternion_slice
                quat_mask = (
                    self.action_loss_mask.to(device=diff.device, dtype=diff.dtype).clone()
                    if self.action_loss_mask is not None
                    else torch.ones(diff.shape[-1], device=diff.device, dtype=diff.dtype)
                )
                quat_mask[qs:qe] = 0.0
                l1_denom = quat_mask.sum() * diff.shape[0] * diff.shape[1]
                action_loss = (diff * quat_mask).sum() / l1_denom

                q_pred = body_pred[..., qs:qe]
                q_target = actions_target[..., qs:qe]
                q_pred = q_pred / q_pred.norm(dim=-1, keepdim=True).clamp_min(1e-6)
                q_target = q_target / q_target.norm(dim=-1, keepdim=True).clamp_min(1e-6)
                # abs() handles the double cover: q and -q denote the same rotation.
                quaternion_loss = (1.0 - (q_pred * q_target).sum(dim=-1).abs()).mean()
                action_loss = action_loss + self.quaternion_loss_weight * quaternion_loss

        out = {"action_loss": action_loss, "per_dim_loss": per_dim_loss}
        if self.hand_head:
            hl = self._hand_losses(hand_logits.float(), examples)
            out["action_loss"] = out["action_loss"] + self.hand_loss_weight * hl["hand_loss"]
            out.update({k: v for k, v in hl.items() if k != "hand_loss"})
            out["hand_loss"] = hl["hand_loss"].detach()
        return out

    @torch.inference_mode()
    def predict_action(
        self,
        examples: List[dict] = None,
        **kwargs: str,
    ) -> np.ndarray:
        """

        Steps:
          1. Resize images to training resolution (if specified)
          2. Encode with QwenVL (hidden states retained)
          6. Return normalized action trajectory

        Returns:
            dict:
                normalized_actions (np.ndarray): Shape [B, T, action_dim], diffusion-sampled normalized actions.
        """
        if type(examples) is not list:
            examples = [examples]
        data_cfg = self.config.datasets.vla_data
        target_size = getattr(data_cfg, "obs_image_size", [224, 224])
        split_indices = getattr(data_cfg, "split_side_by_side_indices", [])
        letterbox_images = bool(getattr(data_cfg, "letterbox_images", False))
        batch_images = []
        for example in examples:
            images = to_pil_preserve(example["image"])
            if not example.get("_images_preprocessed", False):
                images = prepare_multiview_images(
                    images,
                    target_size=target_size,
                    split_side_by_side_indices=split_indices,
                    letterbox=letterbox_images,
                )
            batch_images.append(images)

        instructions = [example["lang"] for example in examples]  # [B, str]
        data_cfg = self.config.datasets.vla_data
        # 손 헤드가 켜져 있으면 hand_prev 는 필수다. 없으면 프롬프트 접두사가 조용히
        # 빠져 학습과 다른 입력이 되고, 오류 없이 성능만 떨어진다.
        if self.hand_head and "hand_prev" not in examples[0]:
            raise ValueError(
                "hand_head=True requires 'hand_prev' in each example — the hand pose "
                "the robot is currently holding, as (on, mode, thumb_flex) per hand. "
                "Training always supplies it from action[t-1]; omitting it at inference "
                "silently drops the prompt suffix the model was trained with."
            )
        hand_prev = (
            [example["hand_prev"] for example in examples] if self.hand_head else None
        )

        # Residual mode needs the state regardless of what the saved config says.
        # config.yaml only records the keys the trainer accessed, and
        # include_state is not one of them, so gating on it here drops the state
        # at inference and the residual has nothing to add - a failure that only
        # shows up at eval time.
        use_state = bool(getattr(data_cfg, "include_state", False)) or self.residual_from_state
        state = (
            [example["state"] for example in examples]
            if use_state and "state" in examples[0]
            else None
        )  # List[ndarray (1, state_dim)] or None

        # Optionally prepend discretised proprioceptive state tokens to each instruction (π₀.5 style).
        # Defaults to on whenever a state is present, as before. Resolved in
        # __init__ from framework.action_model so the choice survives into the
        # saved config.yaml: read from datasets.vla_data it would be absent at
        # inference and silently default back on, adding a prompt prefix the
        # training run never had.
        if state is not None and self.state_in_instruction:
            instructions = self.add_discretized_state_to_instruction(instructions, state)

        # 이전 손 명령을 프롬프트에 붙인다. 관절에서 잔차가 "절대 자세를 픽셀에서
        # 추정"하는 부담을 없앤 것과 같은 역할로, 손은 "지금 무엇을 쥐고 있는가"를
        # 알려 주어 과제를 "유지할지 바꿀지"로 좁힌다. action[t-1] 을 쓰는 이유는
        # action[t] 가 첫 타깃 스텝이라 누출이기 때문이다.
        if hand_prev is not None:
            instructions = [
                instr + self._hand_prev_suffix(hp)
                for instr, hp in zip(instructions, hand_prev)
            ]

        # step 0: add special action token to instruction
        action_tokens = (
            self.action_token * self.chunk_len
        )  # can't add " " between two tokens, otherwise will be tokenized to multiple tokens
        prompt_suffix = f" Please predict the next {self.chunk_len} robot actions: <action>{action_tokens}<action>."
        instructions = [instruction + prompt_suffix for instruction in instructions]

        # Step 1: QWenVL input format
        qwen_inputs = self.qwen_vl_interface.build_qwenvl_inputs(images=batch_images, instructions=instructions)
        with torch.autocast("cuda", dtype=torch.bfloat16):
            qwenvl_outputs = self.qwen_vl_interface(
                **qwen_inputs,
                output_attentions=False,
                output_hidden_states=True,
                return_dict=True,
            )
            # last_hidden_state: [B, seq_len, H]
            last_hidden = qwenvl_outputs.hidden_states[-1]  # [B, L, H]

        # Step 4: Action Expert Forward and Loss
        with torch.autocast("cuda", dtype=torch.float32):
            # Extract action token embeddings as action prediction queries
            input_ids = qwen_inputs.get("input_ids", None)
            action_queries = self._gather_action_token_embeddings(
                last_hidden, input_ids, action_token_id=self.action_token_id
            )  # [B, chunk_len, H]
            pred_actions = self.action_model.predict_action(action_queries)  # (B, chunk_len, action_dim)
            pred_actions = self._apply_state_residual(pred_actions, state)

        normalized_actions = pred_actions.detach().cpu().numpy()
        return {"normalized_actions": normalized_actions}

    def _hand_prev_suffix(self, hand_prev) -> str:
        """이전 손 명령을 프롬프트 접미사 문장으로.

        Args:
            hand_prev: ``(n_hands, 3)`` = (on, mode, thumb_flex). 데이터로더가
                action[t-1] 에서 만든다.

        Returns:
            명령문 뒤에 붙일 문장. 모델은 이걸 읽고 "지금 무엇을 쥐고 있는지" 를
            알게 되어, 과제가 "유지할지 바꿀지" 로 좁혀진다.
        """
        hp = np.asarray(hand_prev, dtype=np.float32).reshape(-1, 3)
        names = ["left", "right"]
        parts = []
        for i in range(hp.shape[0]):
            on, mode, flex = hp[i]
            nm = names[i] if i < len(names) else f"hand{i}"
            if on > 0.5:
                thumb = "thumb in" if flex > 0.5 else "thumb out"
                parts.append(f"{nm} hand gripping shape {int(mode)} with {thumb}")
            else:
                parts.append(f"{nm} hand open")
        return " Hands now: " + ", ".join(parts) + "."

    def _hand_losses(self, hand_logits: torch.Tensor, examples: List[dict]) -> dict:
        """손 분류 loss. on/off 는 가중 BCE, 모드와 엄지굽힘은 on 구간에서만 센다.

        Args:
            hand_logits: ``(B, T, n_hands * 6)`` — 손마다 on(1) + mode(4) + flex(1).
            examples: ``hand_on`` / ``hand_mode`` / ``hand_thumb_flex`` 를 담은 샘플.

        Returns:
            ``{"hand_loss": scalar, ...}`` 진단용 항목 포함. 라벨이 없으면 빈 dict.
        """
        if not all(k in examples[0] for k in ("hand_on", "hand_mode", "hand_thumb_flex")):
            raise ValueError(
                "hand_head=True but the samples carry no hand labels. Set "
                "datasets.vla_data.hand_labels: true and use a DataConfig that "
                "prepends delta index -1 to action_indices."
            )
        dev, dt = hand_logits.device, hand_logits.dtype
        T = hand_logits.shape[1]
        on_t = torch.as_tensor(np.stack([e["hand_on"] for e in examples]), device=dev, dtype=dt)
        md_t = torch.as_tensor(np.stack([e["hand_mode"] for e in examples]), device=dev, dtype=torch.long)
        fx_t = torch.as_tensor(np.stack([e["hand_thumb_flex"] for e in examples]), device=dev, dtype=dt)
        # 라벨은 (B, T_label, n_hands). 타깃 구간 길이에 맞춘다.
        on_t, md_t, fx_t = on_t[:, -T:], md_t[:, -T:], fx_t[:, -T:]

        B = hand_logits.shape[0]
        lg = hand_logits.view(B, T, self.hand_n, self.hand_block)
        on_lg, md_lg, fx_lg = lg[..., 0], lg[..., 1:5], lg[..., 5]

        pw = torch.tensor(self.hand_on_pos_weight, device=dev, dtype=dt)
        on_loss = nn.functional.binary_cross_entropy_with_logits(
            on_lg, on_t, pos_weight=pw
        )
        # 모드와 엄지굽힘은 손이 쥐고 있을 때만 정의된다. off 구간까지 세면
        # 의미 없는 라벨(off 상태의 모드)에 용량을 쓰게 된다.
        m = on_t > 0.5
        if m.any():
            md_loss = nn.functional.cross_entropy(md_lg[m], md_t[m])
            fx_loss = nn.functional.binary_cross_entropy_with_logits(fx_lg[m], fx_t[m])
        else:
            md_loss = on_loss.new_zeros(())
            fx_loss = on_loss.new_zeros(())
        total = on_loss + md_loss + fx_loss
        with torch.no_grad():
            pred_on = on_lg > 0.0
            diag = {
                "hand_on_loss": on_loss.detach(),
                "hand_mode_loss": md_loss.detach(),
                "hand_flex_loss": fx_loss.detach(),
                "hand_on_acc": (pred_on == (on_t > 0.5)).float().mean(),
                "hand_on_rate_gt": (on_t > 0.5).float().mean(),
                "hand_on_rate_pred": pred_on.float().mean(),
            }
        return {"hand_loss": total, **diag}

    def _apply_state_residual(self, pred_actions: torch.Tensor, state) -> torch.Tensor:
        """Add the input state to the head's output, for residual parameterisation.

        Returns ``pred_actions`` untouched unless ``residual_from_state`` is set,
        so the default path is unchanged. Shared by training and inference on
        purpose: if only one of them added the state, the mismatch would not show
        up until evaluation.

        Args:
            pred_actions: ``(B, chunk_len, action_dim)`` head output.
            state: list of ``(1, action_dim)`` arrays, or a tensor shaped
                ``(B, 1, action_dim)`` / ``(B, action_dim)``. The state must be in
                the same normalised layout as the target.

        Returns:
            ``(B, chunk_len, action_dim)``, the state broadcast over the chunk and
            added to the predicted deltas.
        """
        if not self.residual_from_state:
            return pred_actions
        if state is None:
            raise ValueError(
                "residual_from_state=True but no state was provided. Set "
                "datasets.vla_data.include_state: true, and for a future-state "
                "target also state_current_row: true so the state is state[t] "
                "rather than a slice of the target itself."
            )
        s = torch.as_tensor(
            np.array(state) if not isinstance(state, torch.Tensor) else state,
            device=pred_actions.device,
            dtype=pred_actions.dtype,
        )
        if s.dim() == 2:
            s = s.unsqueeze(1)
        if s.dim() != 3 or s.shape[1] != 1:
            raise ValueError(
                f"expected one state row per sample, got shape {tuple(s.shape)}"
            )
        # 손 로짓 구간에는 더하지 않는다. state 를 로짓에 더하는 것은 의미가 없다.
        body = getattr(self, "body_dim", pred_actions.shape[-1])
        if s.shape[-1] != body:
            raise ValueError(
                f"state width {s.shape[-1]} does not match the body width {body} "
                f"(action_dim {pred_actions.shape[-1]}); the residual needs the state "
                "in the same normalised layout as the target."
            )
        if body == pred_actions.shape[-1]:
            return pred_actions + s
        out = pred_actions.clone()
        out[..., :body] = out[..., :body] + s
        return out

    def _gather_action_token_embeddings(
        self,
        last_hidden: torch.Tensor,  # [B, L, H]
        input_ids: torch.Tensor,  # [B, L]
        action_token_id=None,  # Can be int or List[int]
    ) -> torch.Tensor:
        """
        Vectorized batch extraction of action token embeddings:
          - No per-sample for loop
          - Select the last chunk_len action placeholder tokens from each sample
        Args:
            last_hidden: [B, L, H]
            input_ids:   [B, L]
            action_token_id: int or List[int]
        Returns:
            action_queries: [B, chunk_len, H]
        """
        if action_token_id is None:
            raise ValueError("action_token_id must not be None")

        device = input_ids.device
        B, L, H = last_hidden.shape

        # Support multiple ids (e.g., multiple variants)
        if isinstance(action_token_id, (list, tuple, set)):
            id_list = torch.tensor(list(action_token_id), device=device, dtype=input_ids.dtype)
            # torch.isin requires PyTorch >=1.10
            mask = torch.isin(input_ids, id_list)
        else:
            mask = input_ids == action_token_id  # [B, L]

        counts = mask.sum(dim=1)  # [B]
        if (counts < self.chunk_len).any():
            insufficient = (counts < self.chunk_len).nonzero(as_tuple=False).flatten().tolist()
            raise RuntimeError(
                f"The following samples have insufficient action tokens (< {self.chunk_len}): {insufficient} |"
                f" counts={counts.tolist()}"
            )

        # Position indices
        idx = torch.arange(L, device=device).unsqueeze(0).expand(B, L)  # [B, L]
        masked_pos = torch.where(mask, idx, torch.full_like(idx, -1))  # Set non-action positions to -1

        # Take the last chunk_len positions (higher indices = later in sequence)
        # Note: count sufficiency already verified, so -1 won't be incorrectly selected
        topk_pos = masked_pos.topk(k=self.chunk_len, dim=-1).values  # [B, chunk_len] unsorted
        # Sort in temporal order
        selected_pos = topk_pos.sort(dim=-1).values  # [B, chunk_len]

        # Gather
        expanded_index = selected_pos.unsqueeze(-1).expand(-1, -1, H)  # [B, chunk_len, H]
        action_queries = last_hidden.gather(dim=1, index=expanded_index)  # [B, chunk_len, H]
        return action_queries

    # Discretised state → instruction prefix (π₀.5 style); shared with QwenPI_v3.
    add_discretized_state_to_instruction = staticmethod(add_discretized_state_to_instruction)


if __name__ == "__main__":
    import argparse
    import os

    from omegaconf import OmegaConf

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config_yaml",
        type=str,
        default="examples/simBenchmarks/LIBERO/train_files/starvla_cotrain_libero.yaml",
        help="Path to YAML config",
    )
    args, clipargs = parser.parse_known_args()

    if os.getenv("DEBUGPY_ENABLE", "0") == "1":
        import debugpy

        debugpy.listen(("0.0.0.0", 10092))
        print("Rank 0 waiting for debugger attach on port 10092...")
        debugpy.wait_for_client()

    cfg = OmegaConf.load(args.config_yaml)

    model = Qwenvl_OFT(cfg)
    print(model)

    image = Image.fromarray(np.random.randint(0, 255, (224, 224, 3), dtype=np.uint8))
    sample = {
        "action": np.random.uniform(-1, 1, size=(16, 7)).astype(np.float16),
        "image": [image],
        "lang": "This is a fake instruction for testing.",
        "state": np.random.uniform(-1, 1, size=(1, 7)).astype(np.float16),  # chunk, state_dim
    }
    sample2 = sample.copy()
    sample2["lang"] = "Another fake instruction for testing."

    batch = [sample, sample2]
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    forward_output = model(batch)
    action_loss = forward_output["action_loss"]
    print(f"[train] Action Loss (with state): {action_loss.item()}")

    predict_output = model.predict_action(examples=[batch[0]])
    normalized_actions = predict_output["normalized_actions"]
    print(f"[infer] Predicted Action shape: {normalized_actions.shape}")

    # Backward-compat: examples without `state` should still work.
    sample_no_state = {k: v for k, v in sample.items() if k != "state"}
    forward_no_state = model([sample_no_state, sample_no_state])
    print(f"[train] Action Loss (no state): {forward_no_state['action_loss'].item()}")
    predict_no_state = model.predict_action(examples=[sample_no_state])
    print(f"[infer] Predicted Action shape (no state): {predict_no_state['normalized_actions'].shape}")

    print("Finished")
