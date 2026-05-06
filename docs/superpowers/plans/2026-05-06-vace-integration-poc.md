# VACE Integration POC Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add a first-pass VACE subtitle-removal mode that delegates short masked clips to an external VACE worker without installing VACE into the existing service environment.

**Architecture:** Keep the current OCR/ROI detection and audio merge pipeline. Add a focused VACE adapter that builds a mask video, validates external worker configuration, runs VACE via subprocess, and lets `process_video()` use that branch only when `inpaint` is `vace` or `vace-1.3b`.

**Tech Stack:** Python standard library, OpenCV, FastAPI, existing ffmpeg pipeline, external VACE CLI configured by environment variables.

---

### Task 1: VACE Adapter Tests

**Files:**
- Create: `tests/test_vace_adapter.py`

- [x] **Step 1: Write failing tests for VACE configuration and command building**
- [x] **Step 2: Run tests and verify RED**

### Task 2: VACE Adapter Implementation

**Files:**
- Create: `src/inpainting/vace_adapter.py`
- Modify: `src/config.py`

- [x] **Step 1: Add VACE environment configuration**
- [x] **Step 2: Implement adapter**
- [x] **Step 3: Run adapter tests and verify GREEN**

### Task 3: Pipeline and API Wiring

**Files:**
- Modify: `src/pipeline.py`
- Modify: `src/api_server.py`
- Create: `tests/test_pipeline_vace.py`

- [x] **Step 1: Write failing tests for API and pipeline routing**
- [x] **Step 2: Verify RED**
- [x] **Step 3: Wire API and pipeline**
- [x] **Step 4: Verify GREEN**

### Task 4: Documentation and Final Verification

**Files:**
- Modify: `docs/api.md`
- Modify: `README.md`

- [x] **Step 1: Document the experimental mode**
- [x] **Step 2: Run all available verification**
