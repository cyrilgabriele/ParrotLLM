"""Alpaca chat template for ParrotLLM SFT.

Why Alpaca (VL07 slides 31–32, slide 48):

VL07 lists three mainstream chat templates:

    ChatML    (OpenAI):   <|im_start|>role ... <|im_end|>    — new special tokens
    LLaMA-2   (Meta):     [INST]/[/INST] with <<SYS>>/<</SYS>> — new special tokens
    Alpaca    (Stanford): "### Instruction:" / "### Response:" — plain text

VL07 slide 32 "Recommendation for PikoGPT" states explicitly:

    "Alpaca format is the simplest choice. No tokenizer changes required.
     Uses Markdown known syntax."

VL07 slide 32 also flags the invariant that determines quality more than any
other choice:

    "Critical rule: the template used during training MUST match the template
     used during inference."

Project-specific rationale (see docs/post_training/SFT.md §4):

- Our tokenizer is GPT-2 + one added `<|pad|>` token, vocab = 50258. The base
  checkpoint's embedding matrix has learnt exactly those rows. ChatML /
  LLaMA-2 would require adding new rows and relying on SFT alone to train
  them — fragile at our 40M scale and our tight SFT budget.
- Alpaca's plain-text markers tokenise into existing BPE pieces, so zero
  embedding surgery is needed.
- Our handoff to Pair B (DPO) uses the same tokenizer; keeping vocab
  invariant means zero token-ID drift across training stages.

Masking boundary (VL07 slides 15–16):

VL07 slide 15 shows SFT loss computed only on response tokens, with
instruction tokens labelled -100 (the PyTorch `ignore_index` convention for
`nn.CrossEntropyLoss`). VL07 slide 16 quantifies why: without masking,
instruction tokens dominate 30–70% of gradient mass on Short-Q&A / Code-Gen
/ Long-Context / Multi-turn examples. With masking that contribution is
forced to zero, which is what we want — the model already knows how to
predict text from pretraining; SFT teaches it to generate *response* text
conditional on an instruction prefix.

The mask boundary here is the byte offset immediately after the
"### Response:\n" marker. Tokens before → label -100; tokens at and after
(including the EOS) → their own ids.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable


# ── Default Alpaca prompts (Stanford Alpaca repo, Taori et al. 2023) ─────────

ALPACA_PROMPT_NO_INPUT = (
    "Below is an instruction that describes a task. "
    "Write a response that appropriately completes the request.\n\n"
    "### Instruction:\n{instruction}\n\n"
    "### Response:\n"
)

ALPACA_PROMPT_INPUT = (
    "Below is an instruction that describes a task, paired with an input "
    "that provides further context. "
    "Write a response that appropriately completes the request.\n\n"
    "### Instruction:\n{instruction}\n\n"
    "### Input:\n{input}\n\n"
    "### Response:\n"
)


@dataclass(frozen=True)
class AlpacaTemplate:
    """Prompt templates + response boundary.

    The fields are deliberately plain strings so the template is trivially
    serialisable into the checkpoint / config and identical byte-for-byte
    between training and inference (VL07 slide 32's "critical rule").
    """

    prompt_no_input: str = ALPACA_PROMPT_NO_INPUT
    prompt_input: str = ALPACA_PROMPT_INPUT
    response_marker: str = "### Response:\n"

    def render_prompt(self, instruction: str, input_text: str = "") -> str:
        """Return the instruction-side text up to (and including) ``### Response:\\n``.

        This is the string the model *conditions on*; it never contributes
        to the loss. See `render_example` for the full training string.
        """
        if input_text and input_text.strip():
            return self.prompt_input.format(
                instruction=instruction.strip(),
                input=input_text.strip(),
            )
        return self.prompt_no_input.format(instruction=instruction.strip())

    def render_full(
        self,
        instruction: str,
        response: str,
        input_text: str = "",
        eos_token: str = "",
    ) -> str:
        """Return prompt + response + optional EOS, the complete training string.

        VL07 slide 15's diagram shows the sequence `[<user> ... ? <asst> The
        capital ... <eos>]`. Appending the EOS teaches the model to terminate;
        omitting it at inference (`render_prompt`) tells the model to start
        producing a response.
        """
        prompt = self.render_prompt(instruction, input_text)
        return prompt + response.strip() + eos_token


DEFAULT_ALPACA_TEMPLATE = AlpacaTemplate()


def render_example(
    example: dict,
    *,
    template: AlpacaTemplate = DEFAULT_ALPACA_TEMPLATE,
    eos_token: str = "",
) -> tuple[str, str]:
    """Render one ({instruction, input, response}) example.

    Returns:
        (prompt, full_text) — where ``prompt`` ends with the response marker
        and ``full_text = prompt + response + eos_token``. The caller uses
        the byte length of ``prompt`` (or rather the tokeniser-equivalent
        boundary) to construct the -100 label mask in the collator.
    """
    instruction = example.get("instruction", "")
    response = example.get("response", example.get("output", ""))
    input_text = example.get("input", "")

    if not instruction or not response:
        raise ValueError(
            "SFT example must have non-empty 'instruction' and 'response' "
            f"fields. Got keys={list(example.keys())}."
        )

    prompt = template.render_prompt(instruction, input_text)
    full_text = prompt + response.strip() + eos_token
    return prompt, full_text


def normalise_hf_example(raw: dict) -> dict:
    """Map a raw Hugging Face Alpaca-ish example to our internal schema.

    Accepts the common schema variants cited in VL07 slide 17 and Cyril's
    SFT guide (`{messages}` / `{prompt, completion}` / `{instruction, input,
    output}`), and normalises to `{instruction, input, response}`. Raises on
    anything unrecognised so the caller gets a hard failure at data-load
    time rather than silent mis-training.
    """
    if "instruction" in raw and "output" in raw:
        return {
            "instruction": raw["instruction"],
            "input": raw.get("input", ""),
            "response": raw["output"],
        }
    if "instruction" in raw and "response" in raw:
        return {
            "instruction": raw["instruction"],
            "input": raw.get("input", ""),
            "response": raw["response"],
        }
    if "prompt" in raw and "completion" in raw:
        return {
            "instruction": raw["prompt"],
            "input": "",
            "response": raw["completion"],
        }
    if "messages" in raw:
        return _normalise_messages(raw["messages"])
    raise ValueError(f"Unrecognised SFT schema: keys={list(raw.keys())}")


def _normalise_messages(messages: Iterable[dict]) -> dict:
    """Collapse a `{role, content}[]` list into Alpaca's single-turn schema.

    VL07 slide 30 shows the full multi-turn structure; our implementation
    intentionally restricts to the *last* user→assistant exchange because:
    (i) PikoGPT's context is 1024 tokens — long multi-turn histories blow
        the budget;
    (ii) the Alpaca template is single-turn by construction.
    If you need full multi-turn SFT, swap this helper for a ChatML-style
    renderer and add the im_start/im_end special tokens (see VL07 slide 31).
    """
    msgs = list(messages)
    user_msg = next((m for m in reversed(msgs) if m.get("role") == "user"), None)
    asst_msg = next((m for m in reversed(msgs) if m.get("role") == "assistant"), None)
    if user_msg is None or asst_msg is None:
        raise ValueError(
            "messages[] must contain at least one user and one assistant turn."
        )
    system_msg = next((m for m in msgs if m.get("role") == "system"), None)
    instruction = user_msg["content"]
    if system_msg is not None:
        instruction = f"{system_msg['content']}\n\n{instruction}"
    return {
        "instruction": instruction,
        "input": "",
        "response": asst_msg["content"],
    }
