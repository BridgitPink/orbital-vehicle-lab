# Orbital Vehicle Lab

Patch-based overhead vehicle detection pipeline built from the **COWC (Cars Overhead With Context)** dataset and adapted into a **YOLO-compatible training format**.

This project is designed as preparation for overhead imagery vehicle detection research and deployment workflows relevant to remote sensing, edge AI, and small-object detection.

---

## Project Goal

The goal of this repository is to build a practical end-to-end pipeline for:

- working with a legacy overhead imagery dataset
- converting dataset formats for modern training workflows
- preparing a baseline YOLO training pipeline
- later extending the work to edge deployment and full-scene detection

---

## Why COWC

The internship description specifically referenced **COWC**, so this project uses COWC as the starting point for building a modern object-detection workflow.

COWC detection data is distributed as **patch-based samples** rather than a standard modern detection format with full-scene bounding boxes. Because of that, this project converts the detection patch sets into a **YOLO-compatible patch dataset**.

---

## Phase 1: COWC Patch-to-YOLO Conversion

In Phase 1, the COWC detection patch dataset is converted into YOLO format using the following rules:

- **positive patch (`label=1`)** → one centered car bounding box
- **negative patch (`label=0`)** → empty YOLO label file

This produces a dataset that can be used directly with Ultralytics YOLO.

### Important Note

This is a **patch-based YOLO adaptation**, not a full-scene detection benchmark.

Some negative patches may still visually contain cars because the original COWC patch labels are based on **center-target presence**, not necessarily “contains no visible car anywhere in the patch.”

That behavior is inherited from the original dataset design.

---

## Current Dataset Format

```text
data/
├── raw/
│   ├── cowc/
│   └── cowc_detection/
├── processed/
│   └── cowc_yolo_patch/
│       ├── images/
│       │   ├── train/
│       │   ├── val/
│       │   └── test/
│       ├── labels/
│       │   ├── train/
│       │   ├── val/
│       │   └── test/
│       └── dataset.yaml