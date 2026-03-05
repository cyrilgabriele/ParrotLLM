import argparse

from configs import DEFAULT_LANG, PreprocessConfig


def main():
    parser = argparse.ArgumentParser(description="ParrotLLM")
    parser.add_argument("--stage", required=True,
                        choices=["preprocess", "train", "eval", "inference", "chat"])

    # Data args
    parser.add_argument("--dataset-size", default="full", choices=["small", "full", "dummy"])
    # set this default value since we most likely will not change it
    # this due to project restrictions of the TAs/Prof
    parser.add_argument(
        "--lang",
        default=DEFAULT_LANG,
        help="Target language ISO code (default: en for English)",
    )
    parser.add_argument("--data-dir", default="data")

    # Training args
    parser.add_argument("--config", default="configs/default.yaml")
    parser.add_argument("--checkpoint", default=None)

    # Inference args (leaderboard contract)
    parser.add_argument("--prompt", default=None)
    parser.add_argument("--max-tokens", type=int, default=128)
    parser.add_argument("--temperature", type=float, default=None,
                        help="Override sampling temperature (default pulled from config)")
    parser.add_argument("--device", default="auto")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--mock-testing", action="store_true",
                        help="Use a tiny pretrained GPT-2 checkpoint for inference"
                             " instead of a ParrotLLM checkpoint")
    parser.add_argument("--num-workers", default="auto",
                        help="Number of CPU workers for preprocessing (default: auto = all cores)")
    parser.add_argument("--skip-dedup", action="store_true",
                        help="Skip MinHash fuzzy deduplication (faster for quick testing)")
    parser.add_argument("--skip-decontam", action="store_true",
                        help="Skip decontamination index build and filtering (faster for quick testing)")
    parser.add_argument("--filter-mode", default="heuristic",
                        choices=["none", "heuristic", "classifier"],
                        help="Content filtering strategy: none, heuristic (default), or classifier")
    parser.add_argument("--skip-code-filter", action="store_true",
                        help="Skip code/artifact removal phase")
    parser.add_argument("--skip-quality-filter", action="store_true",
                        help="Skip quality/coherence filter phase")
    parser.add_argument("--skip-ellipsis-filter", action="store_true",
                        help="Skip ellipsis-density filter phase (Phase 6.1)")
    parser.add_argument("--target-tokens", type=int, default=None,
                        metavar="N",
                        help="Target token count for the output binary. When set, a seeded "
                             "random subset of OpenWebText is downloaded automatically "
                             "(see --subset-seed). Overrides --dataset-size.")
    parser.add_argument("--subset-seed", type=int, default=42,
                        help="Random seed for the subset shuffle (default: 42). "
                             "Same seed always produces the same subset.")
    parser.add_argument("--topics", nargs="+", default=None,
                        metavar="CLASS[:WEIGHT]",
                        help="Topic classes to keep after RoBERTa classification (Phase 6.2). "
                             "Valid classes: World, Sports, Business, Sci/Tech. "
                             "Optionally append :weight for distribution resampling, "
                             "e.g. --topics Sports:0.4 Business:0.4 World:0.2")
    parser.add_argument("--skip-topic-filter", action="store_true",
                        help="Skip RoBERTa topic classification and resampling (Phase 6.2)")
    parser.add_argument("--leaderboard", action="store_true")

    args = parser.parse_args()

    if args.stage == "preprocess":
        preprocessConfig = PreprocessConfig.from_args(args)
        from src.data.preprocess import run_preprocess
        run_preprocess(preprocessConfig)

    elif args.stage == "train":
        from src.training.trainer import run_train
        run_train(args)

    elif args.stage == "eval":
        from src.eval.perplexity import run_eval
        run_eval(args)

    elif args.stage == "inference":
        from src.eval.inference import run_inference
        run_inference(args)

    elif args.stage == "chat":
        from src.chat.app import run_chat
        run_chat(args)


if __name__ == "__main__":
    main()
