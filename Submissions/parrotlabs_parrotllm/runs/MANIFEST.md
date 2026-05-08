# parrotlabs_parrotllm — checkpoint manifest

The submission checkpoint is a single file expected at:

```
Submissions/parrotlabs_parrotllm/runs/parrotlabs_final.pt
```

It is intentionally **not** tracked in git (458 MB, exceeds GitHub's 100 MB
per-file limit). The canonical copy is hosted publicly on Hugging Face at
[`ParrotLabs/parrotlabs_parrotllm`](https://huggingface.co/ParrotLabs/parrotlabs_parrotllm)
— download it into this directory before running the leaderboard:

```bash
hf download ParrotLabs/parrotlabs_parrotllm parrotlabs_final.pt \
  --local-dir Submissions/parrotlabs_parrotllm/runs
```

(or use `huggingface_hub.hf_hub_download` directly — see the submission README).
Verify the SHA-256 against the value below before running the leaderboard.

## File metadata

- **Filename:** `parrotlabs_final.pt`
- **Version:** v2 (β-down DPO sweep, 2026-05-08)
- **Size:** ~458 MB (479,859,235 bytes)
- **SHA-256:** `0536fd92955600bb216020db6918657df16cccb29ce247fa3b681148ef042ffd`
- **Format:** PyTorch state_dict + config bundle (`torch.load` compatible)

## Provenance

- **Source run:** `runs/posttraining/dpo_continuation_beta001/run_20260508_220825/`
- **Source file:** `checkpoints/best_loss_0p4400_epoch_0000_step_0003100.pt`
- **Stage:** continuation-pair DPO (Plan B), 1 epoch, **β=0.01** (one-tenth of v1's 0.1), lr=2.0e-6
- **Reference:** SFT checkpoint at
  `runs/posttraining/sft_benchmark/run_20260506_010816_sft_lr_5e-07/checkpoints/best_loss_0p9853_epoch_0000_step_0000300.pt`
  (Alpaca template, lr=5e-7, early-stopped at step 300)
- **Best DPO step:** 3100 (training loss 0.4400 from SFT 0.985 from base 6.79)
- **Recipe:** `configs/posttraining/dpo_continuation_beta001.yaml`
- **Design / plan:** `docs/superpowers/specs/2026-05-08-dpo-betadown-design.md`, `docs/superpowers/plans/2026-05-08-dpo-betadown.md`

## Bench results at limit=200

| Benchmark  | v1 (β=0.1) | **v2 (β=0.01)** | Δ |
|---|---|---|---|
| HellaSwag  | 22.00% | **23.50%** | +1.5 |
| WinoGrande | 58.50% | **59.50%** | +1.0 |
| OpenBookQA | 36.50% | **35.50%** | -1.0 |
| LAMBADA    | 36.50% | **38.50%** | +2.0 |
| **Sum**    | 153.50 | **157.00**  | **+3.5** |
| **Average** | 38.38% | **39.25%**  | +0.87 |

Three of four public benchmarks improved; OpenBookQA regressed by 1 point.
The β-down sweep saw monotonically rising LAMBADA across β=0.1 → 0.05 → 0.02 → 0.01
(36.5 → 37.5 → 38.0 → 38.5) and rising sum (153.5 → 153.5 → 154.5 → 157.0).
β=0.05 and β=0.02 results are recorded in `runs/posttraining/dpo_continuation_beta005/`
and `dpo_continuation_beta002/` for reference.

See `Submissions/parrotlabs_parrotllm/README.md` for the full reproduction
recipe and inference contract.
