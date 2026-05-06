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
- **Size:** ~458 MB (479,859,235 bytes)
- **SHA-256:** `1c131cd13b088e875e0705f5a428fffac394005d8f61c947421c2be8c87bf888`
- **Format:** PyTorch state_dict + config bundle (`torch.load` compatible)

## Provenance

- **Source run:** `runs/posttraining/dpo_continuation/run_20260506_023156/`
- **Source file:** `checkpoints/best_loss_0p4400_epoch_0000_step_0002900.pt`
- **Stage:** continuation-pair DPO (Plan B), 1 epoch, β=0.1, lr=2.0e-6
- **Reference:** SFT checkpoint at
  `runs/posttraining/sft_benchmark/run_20260506_010816_sft_lr_5e-07/checkpoints/best_loss_0p9853_epoch_0000_step_0000300.pt`
  (Alpaca template, lr=5e-7, early-stopped at step 300)
- **Best DPO step:** 2900 (training loss 0.4400 from SFT 0.985 from base 6.79)

## Bench results at limit=200

| Benchmark | Score |
|---|---|
| HellaSwag  | 22.00% |
| WinoGrande | 58.50% |
| OpenBookQA | 36.50% |
| LAMBADA    | 36.50% |
| **Average** | **38.38%** |

See `Submissions/parrotlabs_parrotllm/README.md` for the full reproduction
recipe and inference contract.
