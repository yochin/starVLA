"""Unitree G1 WholeBody - SONIC / Dex3 data registry for QwenOFT."""

from starVLA.dataloader.gr00t_lerobot.datasets import ModalityConfig
from starVLA.dataloader.gr00t_lerobot.embodiment_tags import EmbodimentTag
from starVLA.dataloader.gr00t_lerobot.transform.base import ComposedModalityTransform
from starVLA.dataloader.gr00t_lerobot.transform.state_action import StateActionToTensor, StateActionTransform


class UnitreeG1SonicDex3QwenOFTDataConfig:
    embodiment_tag = EmbodimentTag.NEW_EMBODIMENT
    video_keys = ["video.ego_view"]
    state_keys = [
        "state.left_leg",
        "state.right_leg",
        "state.waist",
        "state.left_arm",
        "state.left_hand",
        "state.right_arm",
        "state.right_hand",
        "state.left_wrist_pos",
        "state.left_wrist_abs_quat",
        "state.right_wrist_pos",
        "state.right_wrist_abs_quat",
        "state.root_orientation",
        "state.projected_gravity",
        "state.cpp_rotation_offset",
        "state.init_base_quat",
    ]
    action_keys = [
        "action.motion_token",
        "action.left_hand_joints",
        "action.right_hand_joints",
    ]
    language_keys = ["annotation.human.task_description"]
    observation_indices = [0]
    action_indices = list(range(8))

    state_key_dims = {
        "state.left_leg": 6,
        "state.right_leg": 6,
        "state.waist": 3,
        "state.left_arm": 7,
        "state.left_hand": 7,
        "state.right_arm": 7,
        "state.right_hand": 7,
        "state.left_wrist_pos": 3,
        "state.left_wrist_abs_quat": 4,
        "state.right_wrist_pos": 3,
        "state.right_wrist_abs_quat": 4,
        "state.root_orientation": 4,
        "state.projected_gravity": 3,
        "state.cpp_rotation_offset": 4,
        "state.init_base_quat": 4,
    }
    action_key_dims = {
        "action.motion_token": 64,
        "action.left_hand_joints": 7,
        "action.right_hand_joints": 7,
    }

    def modality_config(self):
        return {
            "video": ModalityConfig(delta_indices=self.observation_indices, modality_keys=self.video_keys),
            "state": ModalityConfig(delta_indices=self.observation_indices, modality_keys=self.state_keys),
            "action": ModalityConfig(delta_indices=self.action_indices, modality_keys=self.action_keys),
            "language": ModalityConfig(delta_indices=self.observation_indices, modality_keys=self.language_keys),
        }

    def transform(self):
        return ComposedModalityTransform(
            transforms=[
                StateActionToTensor(apply_to=self.state_keys),
                StateActionTransform(
                    apply_to=self.state_keys,
                    normalization_modes={key: "q99" for key in self.state_keys},
                ),
                StateActionToTensor(apply_to=self.action_keys),
                StateActionTransform(
                    apply_to=self.action_keys,
                    normalization_modes={key: "q99" for key in self.action_keys},
                ),
            ]
        )


class UnitreeG1DexHandsDirectGR00TDataConfig:
    """Direct 45D G1 joint control with both dexterous hands and three cameras."""

    embodiment_tag = EmbodimentTag.NEW_EMBODIMENT
    video_keys = ["video.front_view", "video.left_wrist_view", "video.right_wrist_view"]
    # Exclude state.g1.action.waist.joint_position because it duplicates
    # current action dimensions rather than representing measured state.
    state_keys = [
        "state.g1.observation.base.angular_velocity",
        "state.g1.observation.base.orientation",
        "state.g1.observation.head.joint_position",
        "state.g1.observation.left_arm.joint_position",
        "state.g1.observation.left_arm.joint_velocity",
        "state.g1.observation.left_hand.joint_position",
        "state.g1.observation.legs.joint_position",
        "state.g1.observation.legs.joint_velocity",
        "state.g1.observation.right_arm.joint_position",
        "state.g1.observation.right_arm.joint_velocity",
        "state.g1.observation.right_hand.joint_position",
        "state.g1.observation.waist.joint_position",
        "state.g1.observation.waist.joint_velocity",
    ]
    action_keys = [
        "action.g1.action.head.joint_position",
        "action.g1.action.left_arm.joint_position",
        "action.g1.action.left_hand.joint_position",
        "action.g1.action.legs.joint_position",
        "action.g1.action.right_arm.joint_position",
        "action.g1.action.right_hand.joint_position",
        "action.g1.action.waist.joint_position",
    ]
    language_keys = ["annotation.human.task_description"]
    observation_indices = [0]
    action_indices = list(range(16))

    state_key_dims = {
        "state.g1.observation.base.angular_velocity": 3,
        "state.g1.observation.base.orientation": 4,
        "state.g1.observation.head.joint_position": 2,
        "state.g1.observation.left_arm.joint_position": 7,
        "state.g1.observation.left_arm.joint_velocity": 7,
        "state.g1.observation.left_hand.joint_position": 7,
        "state.g1.observation.legs.joint_position": 12,
        "state.g1.observation.legs.joint_velocity": 12,
        "state.g1.observation.right_arm.joint_position": 7,
        "state.g1.observation.right_arm.joint_velocity": 7,
        "state.g1.observation.right_hand.joint_position": 7,
        "state.g1.observation.waist.joint_position": 3,
        "state.g1.observation.waist.joint_velocity": 3,
    }
    action_key_dims = {
        "action.g1.action.head.joint_position": 2,
        "action.g1.action.left_arm.joint_position": 7,
        "action.g1.action.left_hand.joint_position": 7,
        "action.g1.action.legs.joint_position": 12,
        "action.g1.action.right_arm.joint_position": 7,
        "action.g1.action.right_hand.joint_position": 7,
        "action.g1.action.waist.joint_position": 3,
    }

    def modality_config(self):
        return {
            "video": ModalityConfig(delta_indices=self.observation_indices, modality_keys=self.video_keys),
            "state": ModalityConfig(delta_indices=self.observation_indices, modality_keys=self.state_keys),
            "action": ModalityConfig(delta_indices=self.action_indices, modality_keys=self.action_keys),
            "language": ModalityConfig(delta_indices=self.observation_indices, modality_keys=self.language_keys),
        }

    def transform(self):
        return ComposedModalityTransform(
            transforms=[
                StateActionToTensor(apply_to=self.state_keys),
                StateActionTransform(
                    apply_to=self.state_keys,
                    normalization_modes={key: "q99" for key in self.state_keys},
                ),
                StateActionToTensor(apply_to=self.action_keys),
                StateActionTransform(
                    apply_to=self.action_keys,
                    normalization_modes={key: "q99" for key in self.action_keys},
                ),
            ]
        )


class UnitreeG1DexHandsStateTargetDataConfig(UnitreeG1DexHandsDirectGR00TDataConfig):
    """Same G1 embodiment, but the regression target is the future 81D state.

    Only the state modality's delta_indices differ from the parent: instead of
    the current frame it covers a 16-step window starting state_lead_frames
    ahead, so the target is state[t+lead .. t+lead+15]. At 30 FPS a lead of 2
    frames is 0.067 s, which matches the measured command-to-encoder lag of the
    arms (2-4 frames).

    Use together with datasets.vla_data.action_target: state. Keep include_state
    false: feeding state[t] while predicting state[t+lead] turns the task into
    near-copying.
    """

    state_lead_frames = 2

    def modality_config(self):
        config = super().modality_config()
        config["state"] = ModalityConfig(
            delta_indices=[self.state_lead_frames + i for i in range(len(self.action_indices))],
            modality_keys=self.state_keys,
        )
        return config


class UnitreeG1DexHandsStateTarget6DDataConfig(UnitreeG1DexHandsStateTargetDataConfig):
    """Future-state target with the base orientation as a 6D rotation, not a quaternion.

    Regressing a quaternion directly is a poor fit for a network: q and -q are the
    same rotation, so the representation is discontinuous and the model has to jump
    at the boundary. In the 81D run this showed up as a base_quat val error 328x the
    persistence baseline. The 6D form is the first two rows of the rotation matrix,
    with the third recovered by cross product, so nothing is lost and the
    representation is continuous (Zhou et al. 2019).

    Two things follow, both handled by the framework rather than by us:
      - the rotation key is normalised with the fixed [-1, 1] statistics for its
        representation instead of per-component dataset q99, so all six components
        share one scale. Per-component scaling is what silently turned the quaternion
        loss into something other than a rotation distance before.
      - unapply() runs the conversion backwards, so deployment still receives a 4D
        quaternion and the 45D command slice is unchanged.

    The state vector grows from 81 to 83 dims and needs no custom loss: plain L1
    covers the rotation along with everything else.
    """

    ROTATION_KEY = "state.g1.observation.base.orientation"

    # What the rotation key is stored as in the parquet. Training reads this
    # from the dataset's modality.json, but deployment rebuilds the metadata
    # from dataset_statistics.json, which does not record it — without this the
    # converter has no source representation and refuses to initialise.
    source_rotation_types = {ROTATION_KEY: "quaternion"}

    def transform(self):
        return ComposedModalityTransform(
            transforms=[
                StateActionToTensor(apply_to=self.state_keys),
                StateActionTransform(
                    apply_to=self.state_keys,
                    # The rotation key must use min_max: the transform requires it for
                    # keys that are converted to another representation, and that path
                    # is what supplies the uniform per-representation statistics.
                    normalization_modes={
                        key: ("min_max" if key == self.ROTATION_KEY else "q99")
                        for key in self.state_keys
                    },
                    target_rotations={self.ROTATION_KEY: "rotation_6d"},
                ),
                StateActionToTensor(apply_to=self.action_keys),
                StateActionTransform(
                    apply_to=self.action_keys,
                    normalization_modes={key: "q99" for key in self.action_keys},
                ),
            ]
        )


ROBOT_TYPE_CONFIG_MAP = {
    "unitree_g1_sonic_dex3": UnitreeG1SonicDex3QwenOFTDataConfig(),
    "unitree_g1_dexhands_direct": UnitreeG1DexHandsDirectGR00TDataConfig(),
    "unitree_g1_dexhands_state_target": UnitreeG1DexHandsStateTargetDataConfig(),
    "unitree_g1_dexhands_state_target_6d": UnitreeG1DexHandsStateTarget6DDataConfig(),
}

DATASET_NAMED_MIXTURES = {
    "unitree_g1_test_sonic": [("test_sonic", 1.0, "unitree_g1_sonic_dex3")],
    "g1_fridge_pick_grapes_train": [
        ("FridgePickGrapes721SepStateObs/train", 1.0, "unitree_g1_dexhands_direct"),
        ("FridgePickGrapes723to724StateObs/train", 1.0, "unitree_g1_dexhands_direct"),
    ],
    "g1_fridge_pick_grapes_val": [
        ("FridgePickGrapes721SepStateObs/val", 1.0, "unitree_g1_dexhands_direct"),
        ("FridgePickGrapes723to724StateObs/val", 1.0, "unitree_g1_dexhands_direct"),
    ],
    # Five fridge manipulation tasks, all recorded at 30 FPS with the same
    # 13-key state / 7-key action layout, so no mixed-FPS handling is needed.
    "g1_fridge5_train": [
        ("FridgeApple/train", 1.0, "unitree_g1_dexhands_direct"),
        ("FridgeGraspLast/train", 1.0, "unitree_g1_dexhands_direct"),
        ("FridgeOnion/train", 1.0, "unitree_g1_dexhands_direct"),
        ("FridgePickCoke/train", 1.0, "unitree_g1_dexhands_direct"),
        ("FridgeTakeCoke/train", 1.0, "unitree_g1_dexhands_direct"),
    ],
    "g1_fridge5_val": [
        ("FridgeApple/val", 1.0, "unitree_g1_dexhands_direct"),
        ("FridgeGraspLast/val", 1.0, "unitree_g1_dexhands_direct"),
        ("FridgeOnion/val", 1.0, "unitree_g1_dexhands_direct"),
        ("FridgePickCoke/val", 1.0, "unitree_g1_dexhands_direct"),
        ("FridgeTakeCoke/val", 1.0, "unitree_g1_dexhands_direct"),
    ],
    # Same five datasets, but read through the state-target DataConfig so the
    # regression target is the future 81D state instead of the 45D action.
    "g1_fridge5_state_train": [
        ("FridgeApple/train", 1.0, "unitree_g1_dexhands_state_target"),
        ("FridgeGraspLast/train", 1.0, "unitree_g1_dexhands_state_target"),
        ("FridgeOnion/train", 1.0, "unitree_g1_dexhands_state_target"),
        ("FridgePickCoke/train", 1.0, "unitree_g1_dexhands_state_target"),
        ("FridgeTakeCoke/train", 1.0, "unitree_g1_dexhands_state_target"),
    ],
    "g1_fridge5_state_val": [
        ("FridgeApple/val", 1.0, "unitree_g1_dexhands_state_target"),
        ("FridgeGraspLast/val", 1.0, "unitree_g1_dexhands_state_target"),
        ("FridgeOnion/val", 1.0, "unitree_g1_dexhands_state_target"),
        ("FridgePickCoke/val", 1.0, "unitree_g1_dexhands_state_target"),
        ("FridgeTakeCoke/val", 1.0, "unitree_g1_dexhands_state_target"),
    ],
    # Same future-state target, but the base orientation is carried as a 6D rotation
    # (83 dims) instead of a quaternion (81).
    "g1_fridge5_state6d_train": [
        ("FridgeApple/train", 1.0, "unitree_g1_dexhands_state_target_6d"),
        ("FridgeGraspLast/train", 1.0, "unitree_g1_dexhands_state_target_6d"),
        ("FridgeOnion/train", 1.0, "unitree_g1_dexhands_state_target_6d"),
        ("FridgePickCoke/train", 1.0, "unitree_g1_dexhands_state_target_6d"),
        ("FridgeTakeCoke/train", 1.0, "unitree_g1_dexhands_state_target_6d"),
    ],
    "g1_fridge5_state6d_val": [
        ("FridgeApple/val", 1.0, "unitree_g1_dexhands_state_target_6d"),
        ("FridgeGraspLast/val", 1.0, "unitree_g1_dexhands_state_target_6d"),
        ("FridgeOnion/val", 1.0, "unitree_g1_dexhands_state_target_6d"),
        ("FridgePickCoke/val", 1.0, "unitree_g1_dexhands_state_target_6d"),
        ("FridgeTakeCoke/val", 1.0, "unitree_g1_dexhands_state_target_6d"),
    ],
    "g1_fridge_picktake_ones_train_mixedFPS_temp": [
        ("FridgePickGrapes721SepStateObs/train", 1.0, "unitree_g1_dexhands_direct"),
        # ("FridgePickGrapes723to724StateObs/train", 1.0, "unitree_g1_dexhands_direct"),
        ("FridgeApple/train", 1.0, "unitree_g1_dexhands_direct"),
        # ("FridgeOnion/train", 1.0, "unitree_g1_dexhands_direct"),
        # ("FridgePickCoke/train", 1.0, "unitree_g1_dexhands_direct"),
        # ("FridgeTakeCoke/train", 1.0, "unitree_g1_dexhands_direct"),        
    ],
    "g1_fridge_picktake_ones_val_mixedFPS_temp": [
        ("FridgePickGrapes721SepStateObs/val", 1.0, "unitree_g1_dexhands_direct"),
        # ("FridgePickGrapes723to724StateObs/val", 1.0, "unitree_g1_dexhands_direct"),
        ("FridgeApple/val", 1.0, "unitree_g1_dexhands_direct"),
        # ("FridgeOnion/val", 1.0, "unitree_g1_dexhands_direct"),
        # ("FridgePickCoke/val", 1.0, "unitree_g1_dexhands_direct"),
        # ("FridgeTakeCoke/val", 1.0, "unitree_g1_dexhands_direct"),        
    ],
}
