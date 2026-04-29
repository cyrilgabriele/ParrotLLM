"""Generate ~800 LAMBADA-style cloze synthetic examples.

LAMBADA prompts are short narratives (2-3 sentences) where the final
word is held out. Format the leaderboard runner sends:

    "...the rest of his time was spent doing the same thing, again and
    again. it was the only thing he could think to do. that, and worry
    about his ____"

It then takes the first whitespace-separated token of the model's
output (after lstrip + punctuation strip) and compares to the gold
word.

Our v6 model emitted dialog snippets ("\\\"!\\nI\\'m") instead of single
words, scoring 0.8% with 45 invalid out of 500. Training on raw cloze
examples should:
  - reduce the invalid count (model learns to emit a single word + EOS)
  - lift correct count (model practises predicting next word in a
    narrative context)

These are entirely synthesized (programmatic templates filling slots).
No corpus reuse — guarantees zero overlap with LAMBADA's underlying
Books3 corpus.

Output: data/synthetic/sft_v7_cloze.jsonl
"""

from __future__ import annotations

import argparse
import json
import random
from pathlib import Path

OUT_PATH = Path("data/synthetic/sft_v7_cloze.jsonl")
SEED = 45
TARGET_TOTAL = 800

# ── Slot vocabularies ──────────────────────────────────────────────────

NAMES_M = ["Tom", "Jack", "Mike", "Dave", "Sam", "Ben", "Chris", "Mark",
           "Paul", "Pete", "Bob", "Steve", "Alex", "Adam", "Eric"]
NAMES_F = ["Anna", "Mary", "Sara", "Kate", "Jane", "Lucy", "Emma",
           "Lisa", "Beth", "Helen", "Carol", "Diana", "Linda"]
PETS = ["dog", "cat", "rabbit", "hamster", "parrot", "puppy", "kitten"]
ROOMS = ["kitchen", "bedroom", "living room", "garage", "garden",
         "basement", "bathroom", "attic"]
FOODS = ["pizza", "soup", "bread", "rice", "pasta", "salad", "sandwich",
         "cookies", "cake", "stew"]
DRINKS = ["coffee", "tea", "water", "milk", "juice", "beer", "wine"]
JOBS = ["teacher", "doctor", "engineer", "writer", "lawyer", "nurse",
        "musician", "chef", "scientist", "artist"]
WEATHER = ["sunny", "cloudy", "rainy", "windy", "warm", "cold", "snowy"]
EMOTIONS = ["happy", "tired", "excited", "nervous", "relaxed", "angry",
            "sad", "calm"]


# ── Cloze templates ─────────────────────────────────────────────────────
# Each template is a function (rng) -> (passage_text, target_word).
# `passage_text` ends just before the held-out word — no trailing space —
# matching the way LAMBADA prompts are sent (the runner adds nothing,
# the model must emit a leading space + the word).

def t_pet_name(rng: random.Random):
    name = rng.choice(NAMES_M + NAMES_F)
    pet = rng.choice(PETS)
    pet_name = rng.choice(NAMES_M + NAMES_F)
    text = (
        f"{name} had owned a {pet} for many years. "
        f"Every morning the {pet} would wait by the door for breakfast. "
        f"{name} loved that {pet} more than anything. "
        f"The name of the {pet} was"
    )
    return text, pet_name


def t_room_object(rng: random.Random):
    name = rng.choice(NAMES_M + NAMES_F)
    room = rng.choice(ROOMS)
    obj = rng.choice(["lamp", "table", "chair", "rug", "sofa", "bookshelf",
                      "mirror", "clock", "vase", "painting"])
    text = (
        f"{name} walked into the {room}. "
        f"The room was quiet and empty except for one piece of furniture. "
        f"In the corner stood a single old"
    )
    return text, obj


def t_food_kitchen(rng: random.Random):
    name = rng.choice(NAMES_M + NAMES_F)
    food = rng.choice(FOODS)
    text = (
        f"{name} entered the kitchen and turned on the oven. "
        f"There were ingredients spread across the counter. "
        f"After two hours of careful preparation, {name} pulled a tray "
        f"out of the oven. The pleasant smell came from the freshly baked"
    )
    return text, food


def t_drink_morning(rng: random.Random):
    name = rng.choice(NAMES_M + NAMES_F)
    drink = rng.choice(DRINKS)
    text = (
        f"{name} woke up early and stumbled into the kitchen. "
        f"The first thing every morning, without fail, was a hot cup of"
    )
    return text, drink


def t_job(rng: random.Random):
    name = rng.choice(NAMES_M + NAMES_F)
    job = rng.choice(JOBS)
    text = (
        f"{name} had studied for many years before earning the qualification. "
        f"After all that hard work, the dream had finally come true. "
        f"{name} was now a fully licensed"
    )
    return text, job


def t_weather(rng: random.Random):
    weather = rng.choice(WEATHER)
    text = (
        f"The forecast had warned about the change in conditions all week. "
        f"By the time everyone gathered outside, the day had turned"
    )
    return text, weather


def t_emotion(rng: random.Random):
    name = rng.choice(NAMES_M + NAMES_F)
    emotion = rng.choice(EMOTIONS)
    text = (
        f"{name} had been waiting for the news for a long time. "
        f"When the email finally arrived with the answer, "
        f"{name} could not help feeling absolutely"
    )
    return text, emotion


def t_animal_baby(rng: random.Random):
    pairs = [("cat", "kitten"), ("dog", "puppy"), ("cow", "calf"),
             ("sheep", "lamb"), ("horse", "foal"), ("pig", "piglet"),
             ("duck", "duckling"), ("goat", "kid"), ("hen", "chick"),
             ("frog", "tadpole"), ("eagle", "eaglet")]
    parent, baby = rng.choice(pairs)
    text = (
        f"At the small farm, the children were excited to see a new arrival. "
        f"The mother {parent} had given birth that morning, and the tiny "
        f"newborn was nestled close to her. The children whispered as they "
        f"watched the little"
    )
    return text, baby


def t_color_object(rng: random.Random):
    pairs = [("the sky", "blue"), ("grass", "green"),
             ("a banana", "yellow"), ("snow", "white"),
             ("an apple", "red"), ("the sun", "yellow"),
             ("a strawberry", "red"), ("a lemon", "yellow"),
             ("a carrot", "orange"), ("a blueberry", "blue"),
             ("an aubergine", "purple"), ("coal", "black"),
             ("a tomato", "red"), ("a pumpkin", "orange")]
    obj, color = rng.choice(pairs)
    text = (
        f"The art teacher held up a card and asked the children to name "
        f"the color. Everyone in the class agreed that {obj} is"
    )
    return text, color


def t_count(rng: random.Random):
    pairs = [("days are in a week", "seven"),
             ("months are in a year", "twelve"),
             ("hours are in a day", "twenty"),  # twenty-four awkward as one token
             ("legs a dog has", "four"),
             ("legs a spider has", "eight"),
             ("legs an insect has", "six"),
             ("wheels a car has", "four"),
             ("wheels a bike has", "two"),
             ("sides a triangle has", "three"),
             ("sides a square has", "four")]
    facts, ans = rng.choice(pairs)
    text = (
        f"The teacher pointed at the chart on the board. "
        f"She asked the students how many {facts}. "
        f"Everyone in the class shouted out the answer at the same time:"
    )
    return text, ans


def t_capital(rng: random.Random):
    pairs = [("France", "Paris"), ("Germany", "Berlin"),
             ("Italy", "Rome"), ("Spain", "Madrid"),
             ("Japan", "Tokyo"), ("China", "Beijing"),
             ("Russia", "Moscow"), ("Egypt", "Cairo"),
             ("Greece", "Athens"), ("Portugal", "Lisbon")]
    country, capital = rng.choice(pairs)
    text = (
        f"At the geography lesson, the teacher pointed at the map. "
        f"She wanted to know the capital of {country}. "
        f"After a moment of silence, the class shouted in unison:"
    )
    return text, capital


TEMPLATES = [
    t_pet_name, t_room_object, t_food_kitchen, t_drink_morning,
    t_job, t_weather, t_emotion, t_animal_baby, t_color_object,
    t_count, t_capital,
]


def _make_row(text: str, target: str) -> dict:
    """Render a cloze row in raw template — no MC markers, no Alpaca.
    The model is trained to emit `' {target}'` (leading space + word)
    then EOS. The leaderboard runner's parser strips whitespace and
    punctuation from the first token and compares case-insensitively."""
    return {"instruction": text, "response": f" {target}"}


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--target", type=int, default=TARGET_TOTAL)
    p.add_argument("--out", type=Path, default=OUT_PATH)
    args = p.parse_args()

    rng = random.Random(SEED)
    rows: list[dict] = []
    while len(rows) < args.target:
        tmpl = rng.choice(TEMPLATES)
        text, target = tmpl(rng)
        rows.append(_make_row(text, target))
    rng.shuffle(rows)

    args.out.parent.mkdir(parents=True, exist_ok=True)
    with args.out.open("w", encoding="utf-8") as fp:
        for ex in rows:
            fp.write(json.dumps(ex, ensure_ascii=False) + "\n")

    from collections import Counter
    targets = Counter(ex["response"].strip() for ex in rows)
    print(f"Wrote {len(rows)} cloze examples to {args.out}")
    print(f"Top 10 target words: {targets.most_common(10)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
