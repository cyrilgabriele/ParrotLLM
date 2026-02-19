import argparse


def main():
    parser = argparse.ArgumentParser(description="ParrotLLM")
    parser.add_argument("--stage", required=True,
                        choices=["preprocess", "train", "eval", "inference", "chat"])

    # Data args
    parser.add_argument("--dataset-size", default="full", choices=["small", "full", "dummy"])
    parser.add_argument("--lang", default="en")
    parser.add_argument("--data-dir", default="data")

    # Training args
    parser.add_argument("--config", default="configs/default.yaml")
    parser.add_argument("--checkpoint", default=None)

    # Inference args (leaderboard contract)
    parser.add_argument("--prompt", default=None)
    parser.add_argument("--max-tokens", type=int, default=128)
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--leaderboard", action="store_true")

    args = parser.parse_args()

    if args.stage == "preprocess":
        from src.data.preprocess import run_preprocess
        run_preprocess(args)

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
