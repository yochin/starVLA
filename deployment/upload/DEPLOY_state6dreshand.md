# G1 fridge-5 — body as a state residual, hands as discrete commands

QwenOFT (Qwen3-VL-4B + MLP head) trained on five G1 fridge tasks: 1,957 episodes,
51.3 h at 30 FPS. The body regresses future state as a residual; the hands are
classified instead of regressed.

**This model's interface differs from the earlier state-target checkpoints.** Read the
two "required" items below before serving it — both fail in ways that produce no error.

## What it predicts

| part | target | loss |
|---|---|---|
| arms (14), legs (12), waist (3), base orientation | `state[t+2 .. t+17]`, as `state[t]` + a predicted delta | L1 |
| **both hands** | the **action** channel, as a class per horizon step | weighted BCE + cross-entropy |
| head | nothing — excluded | — |

Per hand the head emits an on/off logit, four mode logits and a thumb-flexion logit,
appended after the 83 body dims: the raw output is **95 wide**.

Why the hands are classified: 97% of 0.53 s chunks contain no hand change and the rest
are near-full transitions, so L1 settles on the conditional median — "no change" — and
earlier checkpoints scored transition recall of exactly 0.000 while reproducing the
current pose. Why they come from the action channel: a collection bug wrote the command
into the state field for the head and both hands, so their *state* is a command echo in
86% of episodes and a real encoder reading in the rest. The action channel is consistent
across all episodes and is what the robot consumes.

## Required 1 — send `state`, and say it is raw

    "state": raw_state_81d,          # (81,) or (1, 81)
    "state_is_normalized": False,

Training used q99-normalised states. Raw encoder readings span about [-10.3, 7.3]
against [-1.0, 1.0] normalised, a ~2.75x per-dimension difference. Omitting the flag
raises nothing and only degrades the output.

81D layout (drop `[42:45]`, the waist command echo, if the robot reports 84):

    angvel 3 | quat 4 | head 2 | L_arm 7 | L_arm_vel 7 | L_hand 7 | legs 12 |
    legs_vel 12 | R_arm 7 | R_arm_vel 7 | R_hand 7 | waist 3 | waist_vel 3

## Required 2 — send `hand_prev`

    "hand_prev": [[on, mode, thumb_flex],   # left
                  [on, mode, thumb_flex]]   # right, float32 (2, 3)

The pose each hand is currently holding — normally the command you last sent. It is
turned into a prompt suffix ("Hands now: left hand gripping shape 1 with thumb in,
right hand open"), which narrows the task from "what pose" to "hold, or change to
what". This is the categorical counterpart of the residual that fixed the body.

The framework **raises** if this is missing, deliberately: during training it was
always present, so omitting it silently changes the input distribution.

Derive it from your last command: `on` = any of the four finger channels exceeds 0.3 rad,
`mode` = the mode channel rounded, `thumb_flex` = the thumb-flexion channel exceeds 0.5.
Per hand the seven channels are, in order: thumb flexion, mode, thumb rotation, then
index / middle / ring / little. In the 45D action layout that is `action[9..15]` for the
left hand and `action[35..41]` for the right.

## Full payload

    payload = {
        "examples": [{
            "image": [front_view, left_wrist_view, right_wrist_view],  # 3 images, this order
            "lang":  "Pick up the coke from the table and put it into the fridge.",
            "state": raw_state_81d,
            "hand_prev": hand_prev,
        }],
        "unnorm_key": "new_embodiment",
        "state_is_normalized": False,
    }
    # response["data"]["actions"]      -> (B, 16, 81), un-normalised
    # response["data"]["hand_logits"]  -> (B, 16, 12), raw logits

**Send 3 images, not 4.** `front_view` is a side-by-side stereo frame that the model
splits internally (`split_side_by_side_indices: [0]`, `obs_image_size: [224, 224]`,
`letterbox_images: true`, all recorded in `config.yaml`). Do not pre-split it.

## What comes back

The server still returns **81 dims**, not 95. The wrapper strips the hand logits,
un-normalises the body (converting the 6D rotation back to a 4D quaternion), then
writes the predicted hand command into the 81D vector through the hand codebook in
`deployment/model_server/hand_codebook.py`. So existing 81D client code keeps working.

The commandable 45 joint positions are indexed by `ACTION45_FROM_STATE81` in
`examples/G1_Replay/eval_files/state_layout.py`.

## Measured at 80k — read this before serving the model

Open-loop eval, 57,916 windows, stride 100, train+val, against a persistence baseline
(hold the current value). Compared with `state6dres` at 160k, the previous best, on
identical windows and identical eval code:

| | this model (80k) | state6dres (160k) | persistence |
|---|---|---|---|
| overall_clean val (MSE) | 0.00531 (0.98x) | **0.00289 (0.53x)** | 0.00544 |
| left arm / right arm | 0.00347 / 0.00375 | **0.00157 / 0.00152** | 0.00366 / 0.00404 |
| base rotation, geodesic val | 4.05° | **3.14°** | 4.05° |
| hand transition recall val, L / R | **0.081 / 0.144** | 0.000 / 0.000 | 0.000 / 0.000 |
| hand transition precision val, L / R | 0.044 / 0.218 | — | — |

**The body is worse here, not better.** On the arms, the waist and the base rotation this
checkpoint is barely distinguishable from holding the current pose, while the earlier
`state6dres` checkpoint cut the error roughly in half. Two things are confounded — this
run is 80k steps against the other's 160k, and it carries the hand head — and the
80k-vs-80k comparison is still being measured. Until that lands, **prefer
`state6dres` 160k for arm and waist control.**

The hand head did what it was built for in one narrow sense: transition recall is no
longer exactly zero. But precision is 0.044 (left) and 0.218 (right), i.e. 78-96% of the
predicted grasp changes are wrong, and acting on them makes the hands worse in absolute
error than holding still (`part/L_hand` 0.110 vs 0.035 for persistence, a 3.2x
regression). The target to beat was the images-only action model at recall 0.46 /
precision 0.32 on the right hand; this does not beat it.

**So do not command the hands from this checkpoint at the default threshold.** If you want
to use it at all, re-threshold on `hand_logits` for precision first (see below) and
verify on your own data.

## Which joints to command

**Arms and waist. Not the hands at the default threshold** — see the measured numbers
above.

- **Legs (12): do not send.** The real robot's legs are governed by a separate balance
  controller; command-to-encoder correlation is only 0.25-0.55, so predicted leg
  positions fight it.
- **Head (2): do not send.** It is excluded from this model's loss. The head is static
  in 97.8% of 0.53 s windows, so holding the current value is near-optimal.
- **Hands (14): do not send at the default threshold.** This model does predict them, and
  it is the first state-target checkpoint whose transition recall is not exactly 0.000.
  But at `logit > 0` its precision is 0.044 (left) / 0.218 (right), and acting on those
  detections makes the hand error 3.2x worse than holding still. Re-threshold first.

## Tuning the grasp decision without retraining

The on/off decision is taken at `logit > 0`. False positives are expensive: a wrong
grasp on a non-transition frame moves that frame's error from 0.0007 to about 2.0, so
at 50% precision the entire benefit of detecting transitions is cancelled, while at 90%
it survives.

`response["data"]["hand_logits"]` is returned so you can re-threshold on the client.
Logit layout per step: `[L_on, L_mode0..3, L_thumb_flex, R_on, R_mode0..3,
R_thumb_flex]`. Raising the on/off threshold trades recall for precision; pick it on
your own data rather than leaving it at 0.

For reference, the images-only action-target model reached val precision 0.32 /
recall 0.46 on the right hand and 0.26 / 0.32 on the left. Those are the numbers this
head was built to beat, and at the default threshold it does not: 0.218 / 0.144 on the
right, 0.044 / 0.081 on the left. A threshold sweep on val is shipped as
`hand_threshold.json` when the pipeline produced one; `sweep_hand_threshold.py` in the
repo regenerates it from dumped logits without re-running the model.

Note that precision is **not monotonic** in this threshold. A transition is defined as
"predicted state differs from the current state", so pushing the threshold high enough
predicts off everywhere and manufactures transitions wherever the hand is currently
closed. Sweep the whole grid; do not assume higher is safer.

## Caveats

- **State-to-command conversion is an approximation** for the body. The two zero
  references are mostly aligned (offsets ±0.5-2.5°) but `shoulder_roll` measured up to
  **8.7°** apart. Suspect that axis first if an arm looks offset.
- **Hand hardware differs across the training data.** FridgeApple, FridgeOnion,
  FridgePickCoke and FridgeTakeCoke used a BrainCo Revo2; FridgeGraspLast (the grapes
  task) used an Inspire F1. The codebook holds the observed raw command values, which
  were identical per side across all five datasets, but re-derive it if the hands are
  swapped.
- **`config.yaml` under-reports the run.** The trainer records only the keys it
  touched, and three are read inside dataloader workers, so they are absent there and
  present only in `config.full.yaml`: `action_target: state`, `state_current_row: true`,
  `hand_labels: true`. Read `config.full.yaml` if you want to know what was trained.
  Nothing at inference depends on them — the wrapper picks state vs action
  un-normalisation from the emitted width (81 vs 45) — so do not "repair" `config.yaml`.
- **Open-loop metrics have misled repeatedly on this data** — a low-sample checkpoint
  trend, a componentwise quaternion MSE, and the hand aggregate each gave a wrong
  answer. Treat the real-robot result as the authority.
