# Posttraining Plan

## Recommendation

For this project, the best setup is not:

`Base -> SFT -> DPO -> RLHF`

Instead, it should be:

1. `Base -> SFT`
2. Branch from the same SFT checkpoint into:
   - `SFT -> DPO`
   - `SFT -> Reward Model -> RLHF`

This gives us:

- one clean instruction-tuned foundation,
- a fair course-style comparison between DPO and RLHF,
- and, in practice, likely the strongest final model.

If the goal is the best final model, the safest path for this 35.8M model is:

`SFT -> DPO`

Then optionally add a small online RL stage only for narrow, well-rewarded behaviors.

## Why This Fits ParrotLLM

ParrotLLM is a small custom decoder-only model, not yet a Hugging Face `PreTrainedModel`.
That matters because modern post-training tooling such as TRL works best when the model can
be loaded in the standard Hugging Face format.

Current repo realities:

- The core model is a custom `nn.Module` in `src/model/transformer.py`.
- The tokenizer is GPT-2 plus a pad token in `src/utils.py`.
- Chat formatting is currently raw `"User:"` and `"Assistant:"` text in `src/chat/app.py`.
- The main project configuration lives in `configs/default.yaml`.

Because of that, the highest-ROI first step is not choosing between DPO and RLHF. The first
step is making the model plug cleanly into a post-training stack and locking in one stable chat
template.

## Best Practical Strategy

### 1. Make SFT the foundation

SFT should be the first post-training stage.

Why:

- InstructGPT showed that supervised demonstrations already give large gains in usefulness.
- LIMA showed that strong alignment can come from a surprisingly small amount of very clean,
  carefully curated data.
- For a small model, clean SFT data usually gives more return than complicated optimization.

What to do:

- Start with a compact, high-quality instruction dataset.
- Focus on the behaviors you actually want:
  - helpful question answering,
  - rewriting,
  - summarization,
  - simple reasoning,
  - structured outputs such as JSON,
  - concise refusals for unsafe requests,
  - stable assistant style.
- Train with assistant-only or completion-only loss.

Reasonable starting scale for this model:

- roughly `5k-30k` strong examples.

For a model this small, data quality matters more than dataset size inflation.

### 2. Use DPO as the default preference stage

After SFT, DPO should be the default preference optimization method.

Why:

- DPO is much simpler than classical RLHF.
- It is more stable and easier to reproduce.
- It removes the need for online sampling and PPO-style tuning during fine-tuning.
- For small models, this simplicity is a major advantage.

What to do:

- Build explicit prompt preference triples:
  - prompt
  - chosen response
  - rejected response
- Start with vanilla DPO.
- Do not begin with ORPO, KTO, SimPO, or other variants.

The 2026 OXRL study suggests that:

- training paradigm matters,
- but fancy DPO loss variants usually matter much less than people think.

So the right default is plain DPO with good data.

### 3. Do RLHF as a second branch, not the main path

If the course requires RLHF, do it from the same SFT checkpoint and compare it against DPO
fairly.

Recommended branch:

`SFT -> Reward Model -> RLHF`

Practical advice:

- Train the reward model on the same preference pairs used for DPO.
- Keep the prompt distribution as close as possible across DPO and RLHF.
- Report results from both branches on the same held-out evaluation set.

If the goal is the best practical online RL method rather than "classical PPO because the
lecture said RLHF", then I would prefer:

`SFT -> Reward Model -> RLOO`

over:

`SFT -> Reward Model -> PPO`

Why:

- Recent work suggests simpler REINFORCE-style methods can outperform PPO in RLHF-style
  settings.
- RLOO is easier to tune than PPO.
- PPO is still valid for a classical RLHF baseline, but I would not make it the default.

## Concrete Pipeline

## Phase 0: Engineering prerequisites

Before post-training, implement the following:

1. Add a Hugging Face compatible wrapper around ParrotLLM.
2. Add `save_pretrained` / `from_pretrained` support.
3. Add a proper tokenizer + chat template interface.
4. Freeze one prompt format for all post-training stages.
5. Add evaluation prompts and held-out alignment benchmarks.

Without this, SFT/DPO/RLHF will be harder than necessary.

## Phase 1: SFT

Dataset design:

- Use prompt-completion or chat-style examples.
- Keep formatting consistent.
- Prefer short, direct answers over long verbose answers unless explicitly needed.
- Include some multi-turn examples, but do not over-index on long chat if the model is weak.

Training design:

- Full fine-tuning is preferable here.
- Because the model is small, I would not default to LoRA.
- Use completion-only or assistant-only loss.
- Keep early stopping based on held-out instruction-following performance, not just loss.

Expected gain:

- Big improvement in usability, format following, and answer style.

## Phase 2: DPO

Dataset design:

- Same prompt, two responses.
- The chosen response should be better along one or more concrete axes:
  - correctness,
  - usefulness,
  - harmlessness,
  - format compliance,
  - brevity,
  - honesty about uncertainty.

Important rule:

- Rejected responses should be plausible but clearly worse.
- If the rejected sample is too weak, the signal becomes less useful.

Training design:

- Start from the SFT checkpoint.
- Use plain DPO with explicit prompts.
- Keep beta conservative at first.
- Compare DPO against the original SFT checkpoint on the same evaluation suite.

Expected gain:

- Better answer ranking,
- better stylistic alignment,
- fewer awkward or obviously poor responses.

## Phase 3: RLHF

Classical RLHF branch:

1. Train a reward model on preference pairs.
2. Optimize the policy against that reward while keeping KL control to the SFT policy.

Where RL is most worth it:

- structured output validity,
- length control,
- response format obedience,
- simple math or logic with verifiable rewards,
- code tasks with unit-testable outputs,
- refusal policy consistency.

Where RL is less attractive:

- broad open-ended chat where reward quality is noisy,
- tasks where human preferences are inconsistent,
- small models with limited capacity to exploit complex reward shaping.

For this model, RL should be narrow and targeted, not broad and ambitious.

## Data Strategy

For a small model, post-training quality will be dominated by data quality.

### SFT data

Prioritize:

- clean prompts,
- clean answers,
- one stable assistant persona,
- low redundancy,
- high formatting consistency,
- correct labels.

Suggested mix:

- general helpful QA,
- rewriting,
- summarization,
- classification / extraction,
- lightweight reasoning,
- safe refusal examples,
- structured outputs such as JSON.

### Preference data

Prioritize:

- same prompt with meaningful chosen/rejected contrast,
- human or carefully curated synthetic comparisons,
- explicit criteria for why one answer wins,
- no preference pairs where both answers are bad.

Good preference axes:

- factuality,
- honesty,
- directness,
- harmlessness,
- format compliance,
- avoiding unnecessary verbosity.

## Evaluation Plan

Do not evaluate post-training with perplexity alone.

Use a held-out prompt suite and compare:

- Base model
- SFT model
- DPO model
- RLHF model

Metrics to track:

- pairwise preference win rate,
- instruction-following score,
- structured output validity,
- hallucination rate,
- refusal quality,
- verbosity control,
- simple task accuracy on narrow benchmarks.

Recommended evaluation buckets:

1. Helpfulness
2. Harmlessness / refusal quality
3. Honesty / uncertainty handling
4. Format following
5. Short reasoning / task completion

## What I Would Actually Choose

If the goal is the best final model:

- Build one strong SFT checkpoint.
- Train a DPO branch from it.
- Optionally add a small online RL stage only for narrow rewardable tasks.

If the goal is the best course project:

- Build one strong SFT checkpoint.
- Compare:
  - `SFT -> DPO`
  - `SFT -> Reward Model -> RLHF`
- Keep everything else fixed so the comparison is fair.

My expected ranking for this repo is:

1. Best practical result: `SFT -> DPO`
2. Best course comparison setup: `SFT` with parallel `DPO` and `RLHF` branches
3. Least attractive default: jumping directly into PPO-heavy RLHF before strong SFT

## Repo-Specific Next Steps

Short version:

1. Add a Hugging Face export/load path for ParrotLLM.
2. Define one real chat template instead of raw `"User:" / "Assistant:"` formatting.
3. Create a small high-quality SFT dataset.
4. Create preference triples from the same prompt distribution.
5. Train and compare:
   - SFT
   - DPO
   - Reward Model + RLHF
6. Evaluate all three on the same held-out prompt suite.

## Note on the Lecture PDFs

The plan above follows the lecture topics in:

- `docs/VL07_Post_Training_SFT_GH_Edit.pdf`
- `docs/VL08_RLHF_DPO.pdf`

I was not able to reliably machine-extract text from those local PDFs in the sandbox because the
available PDF text tools failed on these browser-exported files. So the detailed recommendations
below are grounded in:

- the lecture topics themselves,
- the local codebase constraints,
- and primary sources from the literature and official tooling docs.

## Sources

- InstructGPT: https://arxiv.org/abs/2203.02155
- DPO: https://arxiv.org/abs/2305.18290
- LIMA: https://arxiv.org/abs/2305.11206
- Llama 2: https://arxiv.org/abs/2307.09288
- Does RLHF Scale? (December 8, 2024): https://arxiv.org/abs/2412.06000
- Back to Basics / RLOO evidence: https://aclanthology.org/2024.acl-long.662/
- OXRL controlled study (March 19, 2026): https://arxiv.org/abs/2603.19335
- TRL overview: https://huggingface.co/docs/trl
- TRL SFTTrainer: https://huggingface.co/docs/trl/sft_trainer
- TRL DPOTrainer: https://huggingface.co/docs/trl/dpo_trainer
- TRL RewardTrainer: https://huggingface.co/docs/trl/reward_trainer
- TRL RLOOTrainer: https://huggingface.co/docs/trl/rloo_trainer
- TRL PPOTrainer: https://huggingface.co/docs/trl/main/ppo_trainer
- SmolLM2-360M-Instruct model card: https://huggingface.co/HuggingFaceTB/SmolLM2-360M-Instruct
