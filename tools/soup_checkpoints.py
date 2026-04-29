"""Average ParrotLLM checkpoint weights into a single soup checkpoint.

Greedy weight-averaging across N checkpoints saved with the same model
config. Trivially safe — does not retrain, does not depend on data, and
only writes to a NEW path. The originals are untouched.

Use cases:
  * Average late SFT trajectory steps (best_step_900, 1000, final) to
    smooth the loss-landscape noise.
  * Cross-soup independent SFT runs (e.g. v6 + v7 finals) to combine
    complementary recipes.

Refusal modes (writes nothing, exits non-zero):
  * Different model configs (different d_model / n_layers / vocab).
  * Different state-dict key sets.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--checkpoints", nargs="+", required=True,
                   help="Paths to .pt checkpoints to average.")
    p.add_argument("--out", required=True,
                   help="Destination .pt path. Refused if it already exists.")
    p.add_argument("--weights", nargs="+", type=float, default=None,
                   help="Optional per-checkpoint weights. Must be same length "
                        "as --checkpoints. Renormalized to sum to 1.")
    p.add_argument("--allow-overwrite", action="store_true")
    args = p.parse_args()

    out_path = Path(args.out)
    if out_path.exists() and not args.allow_overwrite:
        raise SystemExit(f"refusing to overwrite existing {out_path}; "
                         f"pass --allow-overwrite if intended")

    if args.weights is not None:
        if len(args.weights) != len(args.checkpoints):
            raise SystemExit("--weights must match --checkpoints length")
        wsum = sum(args.weights)
        if wsum <= 0:
            raise SystemExit("--weights must sum > 0")
        weights = [w / wsum for w in args.weights]
    else:
        weights = [1.0 / len(args.checkpoints)] * len(args.checkpoints)

    print(f"souping {len(args.checkpoints)} checkpoints into {out_path}")
    for ck, w in zip(args.checkpoints, weights):
        print(f"  weight={w:.4f}  {ck}")

    base = None
    base_keys = None
    base_config = None
    base_meta_extras = None

    for ck_path, w in zip(args.checkpoints, weights):
        ck = torch.load(ck_path, map_location="cpu", weights_only=False)
        if "model" not in ck:
            raise SystemExit(f"{ck_path} has no 'model' state_dict")
        sd = ck["model"]
        cfg = ck.get("config")

        if base is None:
            base = {k: v.float() * w for k, v in sd.items()}
            base_keys = set(sd.keys())
            base_config = cfg
            # Preserve every non-state-dict key from the first checkpoint
            # (training step, args, optimizer state etc. are not meaningful
            # for a soup — we only keep config + a marker).
            base_meta_extras = {k: v for k, v in ck.items()
                                if k not in {"model", "optimizer", "scheduler",
                                             "scaler", "rng_state"}}
        else:
            if set(sd.keys()) != base_keys:
                raise SystemExit(f"{ck_path} state_dict keys differ from "
                                 f"first checkpoint — refusing to soup")
            if cfg is not None and base_config is not None:
                if cfg.get("model") != base_config.get("model"):
                    raise SystemExit(f"{ck_path} model config differs — "
                                     f"refusing to soup")
            for k, v in sd.items():
                base[k] = base[k] + v.float() * w

    # Cast back to the dtype of the first checkpoint to keep file size
    # comparable (typically float32 already, but be safe).
    first = torch.load(args.checkpoints[0], map_location="cpu",
                        weights_only=False)
    target_dtype = next(iter(first["model"].values())).dtype
    base = {k: v.to(target_dtype) for k, v in base.items()}

    out_payload = dict(base_meta_extras or {})
    out_payload["model"] = base
    out_payload["soup_meta"] = {
        "ingredients": list(args.checkpoints),
        "weights": weights,
    }
    if "config" not in out_payload and base_config is not None:
        out_payload["config"] = base_config

    out_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(out_payload, out_path)
    print(f"\nwrote {out_path} ({out_path.stat().st_size / 1e6:.1f} MB)")
    print(f"keys preserved: {sorted(set(out_payload) - {'model', 'soup_meta'})}")
    print(f"soup_meta:\n{json.dumps(out_payload['soup_meta'], indent=2)}")


if __name__ == "__main__":
    main()
