# Hermes Gateway Home-Scoped Restart Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Prevent one Hermes gateway checkout from treating another checkout's gateway PID/lock as its own during update restarts.

**Architecture:** Gateway runtime records become self-identifying by storing `hermes_home`. `get_running_pid()` rejects pid/lock/runtime records that explicitly belong to another `HERMES_HOME`, while preserving legacy records unless their argv clearly points at a different `.hermes*` checkout.

**Tech Stack:** Python 3.11, pytest, systemd user services.

---

### Task 1: Record and Enforce Gateway Home Ownership

**Files:**
- Modify: `gateway/status.py`
- Test: `tests/gateway/test_status.py`

- [ ] Add `hermes_home` to `_build_pid_record()`.
- [ ] Add a home-ownership helper for pid/lock/runtime records.
- [ ] Apply the helper in `get_running_pid()` and `get_runtime_status_running_pid()`.
- [ ] Add regression tests for matching home, mismatched home, legacy argv mismatch, and legacy-compatible records.
- [ ] Run `pytest tests/gateway/test_status.py -q`.
- [ ] Run `pytest tests/gateway/test_restart_service_detection.py tests/gateway/test_restart_drain.py -q`.
- [ ] Commit and push the personal checkout fix to `fork`.
