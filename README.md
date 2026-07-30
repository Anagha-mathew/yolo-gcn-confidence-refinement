# Graph-Based Context-Aware Object Detection for Autonomous Driving

M.Tech dissertation project (BITS Pilani, WILP — AI & ML) exploring whether graph-based
contextual reasoning can improve confidence calibration for object detection in complex
driving scenes, without modifying the underlying detector.

## Problem

Standard object detectors (e.g. YOLO) score each detection independently — they don't
explicitly model relationships between nearby objects, even though object importance and
reliability in driving scenes often depends on surrounding context (a pedestrian near a
moving vehicle vs. a pedestrian standing alone, for example).

## Approach

A frozen **YOLOv8n** detector is used as the baseline. Its detections are converted into a
**scene graph** — each detection becomes a node, with edges built from k-nearest-neighbor
spatial relationships and semantic class-pair weights (e.g. person–car edges weighted
higher than sign–sign edges). A **2-layer Graph Convolutional Network (GCN)** processes
this graph and outputs a learned confidence correction (`gcn_delta`) per node, combined
with YOLO's original confidence in logit space:

```
refined_conf = sigmoid( logit(base_conf) + gcn_delta )
```

Ground-truth labeling uses IoU ≥ 0.5 matching against BDD100K annotations (replacing an
earlier pseudo-label approach that was found to be circular during mid-semester review).

## Pipeline

```
Input image (BDD100K)
   → Frozen YOLOv8n detection
   → Driving-relevant class filtering
   → Detection feature extraction (geometry, confidence, class one-hot)
   → Scene graph construction (k-NN spatial edges + semantic edge weights)
   → Ground-truth IoU-based labeling
   → GCN confidence refinement
   → Evaluation (AP@0.5, mAP@0.5, F1, hard-scene subsets, FP-suppression diagnostics)
```

## Evaluation

- Curated 500-image BDD100K subset (499 valid graph samples) across six scene categories:
  pedestrian–vehicle, dense traffic, occlusion-like, night/low-visibility, traffic-control,
  general driving.
- Five-fold, image-level cross-validation.
- Four ablation variants compared: YOLO baseline, YOLO + GCN, YOLO + semantic rule,
  YOLO + GCN + semantic rule.

### Key results

| Metric | YOLO baseline | YOLO + GCN | YOLO + semantic rule | YOLO + GCN + rule |
|---|---|---|---|---|
| AP@0.5 | 0.2356 | 0.2354 | 0.2358 | 0.2355 |
| mAP@0.5 | 0.1668 | 0.1667 | 0.1669 | 0.1670 |
| F1 @ threshold 0.25 | 0.3203 | 0.3176 | 0.3281 | 0.3263 |

AP/mAP stay close to the YOLO baseline overall — no strong global ranking improvement.
However, the learned GCN suppressed **1,409 false positives** with a negative mean
confidence delta (vs. 0 suppressed by the semantic rule alone), and showed small F1 gains
in the hardest scene subsets (occlusion, night/low-visibility). A regularized variant
(dropout + early stopping) confirmed the near-baseline result isn't driven by overfitting.

## Tech stack

Python, PyTorch, PyTorch Geometric, Ultralytics YOLOv8, scikit-learn, pandas, NumPy,
matplotlib — developed in Google Colab.

## Repository structure

```
notebook/         Full end-to-end pipeline notebook — organized into labeled phases,
                   from environment setup and YOLO inference through GCN training,
                   evaluation, and failure-case visualization
results/          Cross-validation CSVs, hard-scene metrics, FP-suppression diagnostics
docs/             Full dissertation report
README.md
requirements.txt
```

## Limitations & future work

Node features currently encode geometry, confidence, and class label only — no visual
appearance embedding of the detected region. If YOLO misses an object entirely, the graph
has no node to recover it. Planned extensions: ROI-level visual embeddings, object tracking
(DeepSort/ByteTrack) for temporal context across frames, and comparison against GAT-based
attention over neighbors.

## Author

Anagha K M — M.Tech AI & ML, BITS Pilani WILP · [linkedin.com/in/anaghakm](https://linkedin.com/in/anaghakm)
