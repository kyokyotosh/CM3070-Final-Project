# Exercise recogniser

Skeleton-based action recognition for the CM3070 coaching system. Classifies a
two-second window of MediaPipe landmarks as `squat`, `lunge` or `other`.

The model is a published ST-GCN, pre-trained on NTU60 with the COCO-17
keypoint layout, adapted to these three classes. Only the classification head
is new; every layer below it starts from the pre-trained weights, and the
leading blocks are frozen during adaptation.

## The checkpoint

Config `stgcn_8xb16-joint-u100-80e_ntu60-xsub-keypoint-2d`, which declares
`graph_cfg = dict(layout='coco', mode='stgcn_spatial')` and reports 88.95%
top-1 on NTU60 XSub.

```bash
curl -L -o stgcn_ntu60.pth \
  https://download.openmmlab.com/mmaction/v1.0/skeleton/stgcn/stgcn_8xb16-joint-u100-80e_ntu60-xsub-keypoint-2d/stgcn_8xb16-joint-u100-80e_ntu60-xsub-keypoint-2d_20221129-484a394a.pth
```

Confirm it matches this implementation before training anything:

```bash
python3 train.py --mode inspect --checkpoint stgcn_ntu60.pth
```

The report must end with *"every parameter below the head came from the
checkpoint"*. If it lists tensors under `NOT LOADED`, the architecture does
not match and fine-tuning would silently become training from scratch.

## Files

| file | purpose |
|---|---|
| `dataset.py` | loads sessions, remaps 33 landmarks to COCO-17, windows them, two normalisation modes |
| `graph.py` | COCO skeleton adjacency reproducing the checkpoint's own graph |
| `model.py` | ST-GCN matching the published architecture, plus a reporting checkpoint loader |
| `augment.py` | mirror, scale, translate, rotate, keypoint noise and confidence dropout |
| `evaluate.py` | leave-one-session-out cross-validation, confusion matrix, per-class metrics |
| `baseline.py` | hand-crafted geometric features plus a random forest, for comparison |
| `train.py` | entry point: inspect, cross-validate, or produce the final model |

## Requirements

```
pip install torch numpy scikit-learn
```

Torch uses the Metal backend on Apple silicon automatically.

## Running

```bash
# 1. confirm the checkpoint loads completely
python3 train.py --mode inspect --checkpoint stgcn_ntu60.pth

# 2. cross-validate, in the checkpoint's own coordinate frame
python3 train.py --mode cv --checkpoint stgcn_ntu60.pth

# 3. and in the hip-centred frame, to see which transfers better
python3 train.py --mode cv --checkpoint stgcn_ntu60.pth --norm hip

# 4. only then, the model the server loads
python3 train.py --mode final --checkpoint stgcn_ntu60.pth --out recogniser.pt
```

`--freeze N` sets how many of the ten blocks stay fixed; 6 is the default.
Comparing a couple of values is cheap and is exactly the model-selection
evidence the project brief asks for.

## Baseline to beat

Hand-crafted geometry plus a random forest, same windows, same
cross-validation, 14 sessions:

| normalisation | window accuracy | macro F1 | sessions correct |
|---|---|---|---|
| hip-centred | 0.877 | 0.875 | 14/14 |
| image-centred | 0.869 | 0.866 | 14/14 |


