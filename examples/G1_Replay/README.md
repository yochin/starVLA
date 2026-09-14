# G1 Kinematic Replay — 이식 패키지

VLA 정책 예측을 MuJoCo G1으로 재생하고 실제 수집 영상과 나란히 비교하는 파이프라인.
3패널 영상 출력: [실제 헤드캠 | GT(측정 관절값) | 예측(명령)]

## 이 패키지에 없는 것 (직접 준비)

1. **starVLA 체크아웃** — 정책 추론에 필요(`deployment.model_server.policy_wrapper`)
2. **학습 체크포인트** — 본인 런의 `checkpoints/steps_XXXX_pytorch_model.pt`
3. **LeRobot 데이터** — 데이터팩토리 변환본
4. **conda 환경 2개** — 추론용(starVLA+torch), MuJoCo용(mujoco, imageio, Pillow)

## 설치

```bash
# starVLA 체크아웃 안에 그대로 배치
cp -r eval_files assets  <starVLA>/examples/G1_Replay/
```
`eval_files/*.py`는 `parents[3]`을 저장소 루트로 가정하므로
`<starVLA>/examples/<아무이름>/eval_files/` 깊이에 두어야 한다.

## 실행

```bash
cd <starVLA>
export G1_DATA_ROOT=/path/to/LEROBOT_ROOT
export G1_VAL_SETS="myval:mytrain"          # 콤마로 여러 개

# 0) 데이터셋에 평평한 심링크 (train/val 이름 충돌 방지)
cd $G1_DATA_ROOT && ln -sfn "<태스크폴더>/train" mytrain && ln -sfn "<태스크폴더>/val" myval

# 1) modality.json 교정 (126D 복제 결함 우회, 파케이 재작성 없음)
python examples/G1_Replay/eval_files/fix_modality.py $G1_DATA_ROOT/my{train,val}

# 2) 예측 궤적 덤프 (GPU)
CUDA_VISIBLE_DEVICES=0 python examples/G1_Replay/eval_files/dump_pred_traj.py \
    --ckpt <런>/checkpoints/steps_XXXX_pytorch_model.pt \
    --num_chunks 999 --out <런>/pred_traj_full.npz

# 3) 3패널 렌더 (GPU 불필요)
MUJOCO_GL=egl python examples/G1_Replay/eval_files/render_replay_sync.py \
    --npz <런>/pred_traj_full.npz --out_dir <런>/replay_videos_sync
```

## 정량 분석 (선택)

```bash
python .../perjoint_mse.py    --ckpt <ckpt> --out perjoint.json   # 관절 45개별 오차
python .../ee_error_attrib.py --npz perjoint.npz --out ee.json    # 손끝 위치 오차 + 관절 기여도
```

## 함정 (반드시 확인)

- **다리 관절은 SDK 순서** `hip_pitch, hip_roll, hip_yaw, knee, ankle_pitch, ankle_roll`.
  SONIC 설정 파일은 같은 파일 안에 두 순서를 적어놔 혼동하기 쉽다. 데이터로 판정한 결과
  dim0이 무릎과 상관 -0.80, 좌우 동상 +0.84 → hip_pitch가 맞다. 팔·허리는 두 규약 일치.
- **GT 패널은 명령이 아니라 측정값**을 쓴다. SONIC 명령은 50Hz에서 원래 떨린다
  (측정 대비 가속도 RMS 1.8배). 명령 재생은 실기에 없던 떨림을 만든다.
- **손 채널 매핑은 미확정 가설.** 손 7D 중 어느 것이 어느 손가락인지 공식 정의 미수령.
  dim0은 엄지가 아니라 파지 요약값일 가능성이 높다(손가락 채널과 상관 0.987).
  dim1은 관절이 아니라 {0,1,2,3} 이산 파지 모드 셀렉터 — 재생 제외, 학습에서도 회귀 금지.
  **손 모양은 정성 참고용.** 관절각·손끝 위치 수치에는 영향 없음(몸통 29 DoF만 사용).
- **state에서 head+양손 16D 제외** (81D → 65D). 관측이 명령의 비트 단위 복사본이라
  정답이 입력에 새어 지표가 왜곡된다.
- **ground-snap** — 매 프레임 가장 낮은 발을 기준 높이에 맞춰 base z 보정(웅크림 왜곡 방지).
  손끝 위치에 미치는 영향은 0.1cm로 무시 가능함을 확인.

## 에셋 출처

- `assets/g1_meshes/` (36개) — NVIDIA GEAR-SONIC (NVlabs/GR00T-WholeBodyControl)
  `gear_sonic/data/robots/g1/meshes` 중 씬이 참조하는 것만
- `assets/revo2_description/` (38개) — BrainCoTech/revo2_description (커밋 8ca54d2)
- `assets/g1_revo2_scene.xml` — 위 둘을 합성. 손 부착 회전은 공식 Unitree 손 모델의
  손바닥 법선과 대조해 보정 완료(오차 0.0°).

두 저장소 모두 각자의 라이선스를 따른다.
