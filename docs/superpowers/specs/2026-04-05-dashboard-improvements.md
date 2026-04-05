# Dashboard Improvement Notes

**Date:** 2026-04-05
**Status:** Pending implementation

---

## 1. Focus: Live Monitor is the primary view

The Architecture and Run Manager tabs are not very useful in practice. The Live Monitor should be the primary focus of all improvements. Architecture and Run Manager can stay but are lower priority.

---

## 2. Status banner at the very top

A full-width status banner must be the first thing visible — above everything else including the run selector.

- **When no problems:** green banner, e.g. `✅  No errors or malfunctions detected`
- **When problems exist:** red/yellow banner listing alerts prominently, e.g. `🔴  GRAD_EXPLOSION — Grad norm 14.2 in last 3 steps. Reduce LR.`
- This replaces the current hidden/collapsed alerts row — alerts must be immediately obvious, not tucked away
- Multiple alerts stack vertically in the banner

---

## 3. Progress section — clearer and with historical context

Current progress is a single small textbox. Needs to be more prominent and informative:

- **Larger text / dedicated panel**, not a tiny textbox
- Show current values clearly: Step, Train Loss, Val Loss, LR, Grad Norm, Tok/s, Best Step
- **Show historical deltas**: e.g. "Loss Δ −0.023 last 10 steps" or "Val Loss improved 0.12 since best checkpoint"
- ETA must show hours + minutes + seconds: `~2h 14m 32s` (not just `~2h 14m`)
- Progress bar showing `step / max_steps` as a visual bar, not just text

---

## 4. Graph layout — reorder and resize

Most actively changing metrics go on top, slower/less critical below.

**Top row (most active, larger):**
1. Train & Val Loss
2. Tokens/sec
3. LR & Grad Norm (twin axis)

**Bottom row (contextual, can be smaller):**
4. Validation Perplexity
5. Generalization Gap
6. (empty or grad norm standalone if split from LR)

Graphs should be **larger overall** — increase figure height and use more of the available screen width. Currently they are too small to read comfortably.

---

## 5. Architecture tab — improve information hierarchy

Architecture summary should mirror the output printed to console when training starts. Key information at the top so it is easy to verify at a glance:

```
Total params:     35,763,840
Trainable params: 35,763,840
Vocab size:       50,257
Layers:           16
Heads:            8
d_model:          320
FFN dim (d_ff):   854
Context length:   1024
```

Most important fields first: total params, trainable params, vocab size, layers, heads. Less important fields (d_ff, context length) below.

**Configs must also be shown** in the Architecture tab — not just the model architecture but the full training config (batch size, max steps, gradient accumulation steps, learning rate, etc.). These are currently missing entirely and are important for verifying a run is configured correctly.

Suggested layout:
- Section 1: Model architecture (params, layers, etc.)
- Section 2: Training config (lr, batch size, max_steps, context_length, grad accumulation, etc.)
- Section 3: Raw JSON accordion (already exists, keep it)

---

## Summary of changes by file

| File | Change |
|------|--------|
| `src/dashboard/app.py` | Move alerts to top as full-width banner; enlarge progress section with deltas + progress bar; fix ETA to h/m/s |
| `src/dashboard/plots.py` | Reorder panels: Loss, Tok/s, LR+Grad on top row; PPL, Gap below; increase figure size |
| `src/dashboard/app.py` (Architecture tab) | Show params-first layout matching training startup output; add training config section |
