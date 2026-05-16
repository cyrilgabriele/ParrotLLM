"""Generate ~1000 synthetic raw-format MC examples for SFT v6.

Each row is a programmatic factual question with exactly one verifiably-
correct answer letter. Format (byte-identical to what the leaderboard
runner sends at inference):

    {
      "instruction": "Context: What is the capital of France?\\nA) Madrid\\nB) Paris\\nC) Rome\\nD) Berlin\\nAnswer:",
      "response": " B"
    }

Categories:

  - capitals (50)              4-choice
  - colors (50)                4-choice
  - arithmetic (200)           4-choice (add / sub / mul)
  - counts (50)                4-choice (days/months/legs/etc.)
  - day-month (30)             4-choice
  - animal-class (50)          4-choice (mammal/bird/fish/reptile)
  - chemistry (50)             4-choice (H2O = water etc.)
  - synonyms (200)             4-choice
  - winogrande-style 2-choice (300)

Output: data/synthetic/sft_v6_programmatic.jsonl  (~1000 rows)

Decontamination is handled downstream by build_sft_datasets() at training
time; nothing here references any benchmark test split.
"""

from __future__ import annotations

import json
import random
from pathlib import Path

OUT_PATH = Path("data/synthetic/sft_v6_programmatic.jsonl")
SEED = 42

# ── Source facts ────────────────────────────────────────────────────────

CAPITALS = [
    ("France", "Paris"), ("Germany", "Berlin"), ("Italy", "Rome"),
    ("Spain", "Madrid"), ("Portugal", "Lisbon"), ("Greece", "Athens"),
    ("Poland", "Warsaw"), ("Russia", "Moscow"), ("Ukraine", "Kyiv"),
    ("Sweden", "Stockholm"), ("Norway", "Oslo"), ("Denmark", "Copenhagen"),
    ("Finland", "Helsinki"), ("Netherlands", "Amsterdam"), ("Belgium", "Brussels"),
    ("Austria", "Vienna"), ("Switzerland", "Bern"), ("Hungary", "Budapest"),
    ("Czech Republic", "Prague"), ("Romania", "Bucharest"), ("Bulgaria", "Sofia"),
    ("Turkey", "Ankara"), ("Egypt", "Cairo"), ("Morocco", "Rabat"),
    ("South Africa", "Pretoria"), ("Kenya", "Nairobi"), ("Nigeria", "Abuja"),
    ("Ethiopia", "Addis Ababa"), ("Japan", "Tokyo"), ("China", "Beijing"),
    ("South Korea", "Seoul"), ("Vietnam", "Hanoi"), ("Thailand", "Bangkok"),
    ("Indonesia", "Jakarta"), ("Philippines", "Manila"), ("India", "New Delhi"),
    ("Pakistan", "Islamabad"), ("Bangladesh", "Dhaka"), ("Iran", "Tehran"),
    ("Iraq", "Baghdad"), ("Saudi Arabia", "Riyadh"), ("Israel", "Jerusalem"),
    ("Australia", "Canberra"), ("New Zealand", "Wellington"),
    ("Canada", "Ottawa"), ("Mexico", "Mexico City"), ("Argentina", "Buenos Aires"),
    ("Brazil", "Brasilia"), ("Chile", "Santiago"), ("Cuba", "Havana"),
    ("Ireland", "Dublin"), ("Iceland", "Reykjavik"),
]

COLORS = [
    ("the sky on a clear day", "blue"),
    ("grass", "green"), ("a ripe banana", "yellow"),
    ("snow", "white"), ("coal", "black"), ("a ripe tomato", "red"),
    ("an orange (the fruit)", "orange"), ("a lemon", "yellow"),
    ("a typical leaf in summer", "green"), ("milk", "white"),
    ("the sun (as commonly drawn)", "yellow"),
    ("a stop sign", "red"), ("a dollar bill", "green"),
    ("a typical taxi in New York", "yellow"),
    ("a polar bear", "white"), ("a raven", "black"),
    ("a flamingo", "pink"), ("a typical crow", "black"),
    ("a strawberry", "red"), ("a blueberry", "blue"),
    ("a pumpkin", "orange"), ("an eggplant", "purple"),
    ("a typical lime", "green"), ("a typical cucumber", "green"),
    ("a peach", "orange"), ("rust", "brown"),
    ("a chocolate bar", "brown"), ("ash", "gray"),
    ("a dolphin", "gray"), ("an elephant", "gray"),
    ("a typical mouse (animal)", "gray"), ("a giraffe's spots", "brown"),
    ("a brick", "red"), ("the inside of a watermelon", "red"),
    ("the inside of a kiwi fruit", "green"), ("the inside of a banana", "white"),
    ("typical printer paper", "white"), ("the ocean (deep)", "blue"),
    ("a clear glass of water", "clear"), ("clouds on a sunny day", "white"),
    ("a fire truck (typical)", "red"), ("a school bus (typical USA)", "yellow"),
    ("a leprechaun's hat (traditional)", "green"),
    ("Santa's suit (traditional)", "red"),
    ("a panda's body", "white"), ("a panda's eye patches", "black"),
    ("the night sky", "black"), ("a typical sunflower's center", "brown"),
    ("a typical sunflower's petals", "yellow"),
    ("a typical traffic-light go signal", "green"),
    ("a typical traffic-light stop signal", "red"),
]

# Animal -> class
ANIMALS = [
    ("a dog", "mammal"), ("a cat", "mammal"), ("a horse", "mammal"),
    ("a cow", "mammal"), ("a sheep", "mammal"), ("a pig", "mammal"),
    ("a goat", "mammal"), ("a rabbit", "mammal"), ("a deer", "mammal"),
    ("an elephant", "mammal"), ("a tiger", "mammal"), ("a lion", "mammal"),
    ("a bear", "mammal"), ("a whale", "mammal"), ("a dolphin", "mammal"),
    ("a bat", "mammal"), ("a kangaroo", "mammal"),
    ("a chicken", "bird"), ("a duck", "bird"), ("an eagle", "bird"),
    ("a parrot", "bird"), ("a penguin", "bird"), ("an owl", "bird"),
    ("a sparrow", "bird"), ("a swan", "bird"), ("a flamingo", "bird"),
    ("a salmon", "fish"), ("a tuna", "fish"), ("a shark", "fish"),
    ("a goldfish", "fish"), ("a trout", "fish"), ("a cod", "fish"),
    ("a bass", "fish"), ("a herring", "fish"),
    ("a snake", "reptile"), ("a lizard", "reptile"), ("a turtle", "reptile"),
    ("a crocodile", "reptile"), ("an alligator", "reptile"),
    ("a gecko", "reptile"), ("an iguana", "reptile"), ("a tortoise", "reptile"),
    ("a frog", "amphibian"), ("a toad", "amphibian"),
    ("a salamander", "amphibian"), ("a newt", "amphibian"),
    ("a butterfly", "insect"), ("a bee", "insect"), ("an ant", "insect"),
    ("a beetle", "insect"), ("a wasp", "insect"),
]

CHEMISTRY = [
    ("H2O", "water"), ("NaCl", "table salt"),
    ("CO2", "carbon dioxide"), ("O2", "oxygen"), ("N2", "nitrogen"),
    ("H2", "hydrogen"), ("CH4", "methane"), ("C2H5OH", "ethanol"),
    ("HCl", "hydrochloric acid"), ("NaOH", "sodium hydroxide"),
    ("CaCO3", "calcium carbonate"), ("Fe2O3", "iron oxide (rust)"),
    ("SO2", "sulphur dioxide"), ("NH3", "ammonia"),
    ("H2SO4", "sulphuric acid"), ("HNO3", "nitric acid"),
    ("CO", "carbon monoxide"), ("O3", "ozone"),
    ("Au", "gold"), ("Ag", "silver"), ("Fe", "iron"), ("Cu", "copper"),
    ("Pb", "lead"), ("Sn", "tin"), ("Hg", "mercury"), ("Zn", "zinc"),
    ("Al", "aluminium"), ("Mg", "magnesium"), ("Ca", "calcium"),
    ("K", "potassium"), ("Na", "sodium"),
    ("He", "helium"), ("Ne", "neon"), ("Ar", "argon"),
    ("Cl", "chlorine"), ("F", "fluorine"), ("Br", "bromine"), ("I", "iodine"),
    ("S", "sulphur"), ("P", "phosphorus"), ("Si", "silicon"),
    ("C", "carbon"), ("N", "nitrogen"), ("O", "oxygen"),
]

# 200 simple synonym/antonym pairs
SYNONYM_PAIRS = [
    ("happy", ["joyful", "sad", "angry", "tired"], 0),
    ("big", ["large", "small", "tiny", "narrow"], 0),
    ("small", ["tiny", "huge", "loud", "fast"], 0),
    ("fast", ["quick", "slow", "loud", "tall"], 0),
    ("slow", ["sluggish", "quick", "smart", "loud"], 0),
    ("smart", ["clever", "stupid", "tall", "wide"], 0),
    ("hot", ["warm", "cold", "wet", "dry"], 0),
    ("cold", ["chilly", "hot", "loud", "soft"], 0),
    ("bright", ["luminous", "dark", "soft", "loud"], 0),
    ("dark", ["dim", "bright", "fast", "slow"], 0),
    ("loud", ["noisy", "quiet", "bright", "small"], 0),
    ("quiet", ["silent", "loud", "fast", "wide"], 0),
    ("strong", ["powerful", "weak", "soft", "thin"], 0),
    ("weak", ["feeble", "strong", "loud", "tall"], 0),
    ("rich", ["wealthy", "poor", "cold", "loud"], 0),
    ("poor", ["destitute", "rich", "fast", "tall"], 0),
    ("brave", ["courageous", "cowardly", "tired", "small"], 0),
    ("kind", ["gentle", "cruel", "fast", "wide"], 0),
    ("cruel", ["harsh", "kind", "warm", "bright"], 0),
    ("beautiful", ["lovely", "ugly", "fast", "loud"], 0),
    ("ugly", ["unattractive", "beautiful", "fast", "warm"], 0),
    ("easy", ["simple", "hard", "loud", "wet"], 0),
    ("hard", ["difficult", "easy", "warm", "fast"], 0),
    ("rare", ["uncommon", "common", "fast", "warm"], 0),
    ("common", ["frequent", "rare", "loud", "tall"], 0),
    ("begin", ["start", "end", "stop", "rest"], 0),
    ("end", ["finish", "begin", "throw", "swim"], 0),
    ("buy", ["purchase", "sell", "throw", "carry"], 0),
    ("sell", ["vend", "buy", "carry", "swim"], 0),
    ("speak", ["talk", "listen", "swim", "build"], 0),
    ("listen", ["hear", "speak", "swim", "throw"], 0),
    ("look", ["see", "ignore", "throw", "swim"], 0),
    ("walk", ["stroll", "run", "swim", "fly"], 0),
    ("run", ["sprint", "walk", "swim", "stand"], 0),
    ("eat", ["consume", "drink", "throw", "carry"], 0),
    ("drink", ["sip", "eat", "throw", "carry"], 0),
    ("sleep", ["rest", "wake", "run", "build"], 0),
    ("build", ["construct", "destroy", "throw", "carry"], 0),
    ("break", ["shatter", "fix", "carry", "swim"], 0),
    ("teach", ["instruct", "learn", "carry", "swim"], 0),
    ("learn", ["study", "teach", "carry", "throw"], 0),
    ("ask", ["inquire", "answer", "throw", "swim"], 0),
    ("answer", ["reply", "ask", "throw", "carry"], 0),
    ("give", ["donate", "take", "throw", "carry"], 0),
    ("take", ["grab", "give", "swim", "fly"], 0),
    ("open", ["unlock", "close", "swim", "carry"], 0),
    ("close", ["shut", "open", "fly", "swim"], 0),
    ("start", ["begin", "stop", "throw", "carry"], 0),
    ("stop", ["halt", "go", "throw", "swim"], 0),
    ("clean", ["spotless", "dirty", "tall", "loud"], 0),
    ("dirty", ["filthy", "clean", "fast", "loud"], 0),
    ("dry", ["arid", "wet", "loud", "fast"], 0),
    ("wet", ["damp", "dry", "loud", "fast"], 0),
    ("safe", ["secure", "dangerous", "loud", "fast"], 0),
    ("dangerous", ["risky", "safe", "loud", "fast"], 0),
    ("true", ["correct", "false", "tall", "loud"], 0),
    ("false", ["incorrect", "true", "tall", "loud"], 0),
    ("near", ["close", "far", "tall", "loud"], 0),
    ("far", ["distant", "near", "tall", "loud"], 0),
    ("up", ["above", "down", "left", "right"], 0),
    ("down", ["below", "up", "left", "right"], 0),
    ("inside", ["within", "outside", "above", "below"], 0),
    ("outside", ["beyond", "inside", "above", "below"], 0),
    ("first", ["initial", "last", "second", "third"], 0),
    ("last", ["final", "first", "second", "third"], 0),
    ("old", ["aged", "young", "fast", "loud"], 0),
    ("young", ["youthful", "old", "fast", "loud"], 0),
    ("new", ["recent", "old", "fast", "loud"], 0),
    ("full", ["loaded", "empty", "fast", "loud"], 0),
    ("empty", ["vacant", "full", "fast", "loud"], 0),
    ("heavy", ["weighty", "light", "fast", "loud"], 0),
    ("light", ["weightless", "heavy", "fast", "loud"], 0),
    ("tall", ["high", "short", "fast", "loud"], 0),
    ("short", ["brief", "tall", "fast", "loud"], 0),
    ("wide", ["broad", "narrow", "fast", "loud"], 0),
    ("narrow", ["thin", "wide", "fast", "loud"], 0),
    ("thick", ["dense", "thin", "fast", "loud"], 0),
    ("thin", ["slim", "thick", "fast", "loud"], 0),
    ("rough", ["coarse", "smooth", "fast", "loud"], 0),
    ("smooth", ["even", "rough", "fast", "loud"], 0),
    ("sharp", ["keen", "dull", "fast", "loud"], 0),
    ("dull", ["blunt", "sharp", "fast", "loud"], 0),
    ("hard (texture)", ["firm", "soft", "fast", "loud"], 0),
    ("soft", ["pliable", "hard", "fast", "loud"], 0),
    ("alive", ["living", "dead", "fast", "loud"], 0),
    ("dead", ["deceased", "alive", "fast", "loud"], 0),
    ("real", ["genuine", "fake", "fast", "loud"], 0),
    ("fake", ["false", "real", "fast", "loud"], 0),
    ("free", ["liberated", "captive", "fast", "loud"], 0),
    ("busy", ["occupied", "idle", "fast", "loud"], 0),
    ("idle", ["inactive", "busy", "fast", "loud"], 0),
    ("calm", ["peaceful", "agitated", "fast", "loud"], 0),
    ("angry", ["furious", "calm", "fast", "loud"], 0),
    ("ready", ["prepared", "unprepared", "fast", "loud"], 0),
    ("hungry", ["starving", "full", "fast", "loud"], 0),
    ("thirsty", ["parched", "quenched", "fast", "loud"], 0),
    ("tired", ["exhausted", "energetic", "fast", "loud"], 0),
    ("simple", ["plain", "complex", "fast", "loud"], 0),
    ("complex", ["complicated", "simple", "fast", "loud"], 0),
    ("important", ["vital", "trivial", "fast", "loud"], 0),
    ("trivial", ["minor", "important", "fast", "loud"], 0),
    ("brief", ["short", "lengthy", "fast", "loud"], 0),
    ("lengthy", ["long", "brief", "fast", "loud"], 0),
]


def make_4_choice(question: str, correct: str, distractors: list[str], rng: random.Random) -> dict:
    """Return one synthetic raw-format MC example.

    Choices are shuffled deterministically by ``rng``; the answer letter
    is whichever letter the correct answer ends up at.
    """
    options = [correct] + list(distractors)[:3]
    rng.shuffle(options)
    correct_idx = options.index(correct)
    letter = "ABCD"[correct_idx]
    instruction = (
        f"Context: {question}\n"
        f"A) {options[0]}\n"
        f"B) {options[1]}\n"
        f"C) {options[2]}\n"
        f"D) {options[3]}\n"
        f"Answer:"
    )
    response = f" {letter}"
    return {"instruction": instruction, "response": response}


def make_2_choice(context: str, correct: str, wrong: str, rng: random.Random) -> dict:
    """Return one Winogrande-style 2-choice raw example."""
    if rng.random() < 0.5:
        a, b = correct, wrong
        letter = "A"
    else:
        a, b = wrong, correct
        letter = "B"
    instruction = (
        f"Context: {context}\n"
        f"A) {a}\n"
        f"B) {b}\n"
        f"Answer:"
    )
    return {"instruction": instruction, "response": f" {letter}"}


def gen_capitals(rng: random.Random) -> list[dict]:
    out: list[dict] = []
    cities = [c for _, c in CAPITALS]
    for country, city in CAPITALS:
        distractors = rng.sample([c for c in cities if c != city], 3)
        out.append(make_4_choice(
            f"What is the capital of {country}?",
            city, distractors, rng,
        ))
    return out


def gen_colors(rng: random.Random) -> list[dict]:
    out: list[dict] = []
    palette = ["red", "orange", "yellow", "green", "blue", "purple",
               "pink", "brown", "black", "white", "gray", "clear"]
    for thing, color in COLORS:
        distractors = rng.sample([c for c in palette if c != color], 3)
        out.append(make_4_choice(
            f"What color is {thing}?", color, distractors, rng,
        ))
    return out


def gen_arithmetic(rng: random.Random, n: int = 200) -> list[dict]:
    out: list[dict] = []
    for _ in range(n):
        op = rng.choice(["+", "-", "×"])
        if op == "+":
            a, b = rng.randint(1, 50), rng.randint(1, 50)
            ans = a + b
        elif op == "-":
            a, b = rng.randint(1, 50), rng.randint(1, 50)
            if a < b:
                a, b = b, a
            ans = a - b
        else:
            a, b = rng.randint(1, 12), rng.randint(1, 12)
            ans = a * b
        # Distractors: nearby wrong answers
        cand = set()
        while len(cand) < 3:
            delta = rng.choice([-3, -2, -1, 1, 2, 3, 5, -5])
            v = ans + delta
            if v != ans and v >= 0:
                cand.add(v)
        distractors = [str(x) for x in cand]
        sym = "*" if op == "×" else op
        out.append(make_4_choice(
            f"What is {a} {sym} {b}?", str(ans), distractors, rng,
        ))
    return out


def gen_counts(rng: random.Random) -> list[dict]:
    facts = [
        ("How many days are in a week?", "7", ["5", "6", "8", "10", "12"]),
        ("How many months are in a year?", "12", ["10", "11", "13", "8", "6"]),
        ("How many hours are in a day?", "24", ["12", "20", "30", "48"]),
        ("How many minutes are in an hour?", "60", ["30", "45", "90", "120"]),
        ("How many seconds are in a minute?", "60", ["30", "50", "100", "120"]),
        ("How many legs does a dog have?", "4", ["2", "3", "6", "8"]),
        ("How many legs does a spider have?", "8", ["4", "6", "10", "12"]),
        ("How many legs does an insect have?", "6", ["4", "8", "10", "12"]),
        ("How many wheels does a typical car have?", "4", ["2", "3", "6", "8"]),
        ("How many wheels does a bicycle have?", "2", ["1", "3", "4", "6"]),
        ("How many sides does a triangle have?", "3", ["2", "4", "5", "6"]),
        ("How many sides does a square have?", "4", ["3", "5", "6", "8"]),
        ("How many sides does a hexagon have?", "6", ["4", "5", "7", "8"]),
        ("How many sides does a pentagon have?", "5", ["3", "4", "6", "8"]),
        ("How many planets are in the solar system (current count)?", "8",
         ["7", "9", "10", "12"]),
        ("How many continents are there?", "7", ["5", "6", "8", "9"]),
        ("How many oceans are there (commonly five)?", "5", ["3", "4", "6", "7"]),
        ("How many days are in February (non-leap year)?", "28",
         ["29", "30", "31", "27"]),
        ("How many days are in January?", "31", ["28", "29", "30", "32"]),
        ("How many days are in June?", "30", ["28", "29", "31", "32"]),
        ("How many days are typically in a leap year?", "366",
         ["365", "364", "367", "360"]),
        ("How many days in a typical year?", "365", ["360", "364", "366", "368"]),
        ("How many letters are in the English alphabet?", "26",
         ["24", "25", "27", "30"]),
        ("How many strings does a standard guitar have?", "6",
         ["4", "5", "7", "8"]),
        ("How many keys does a standard piano have?", "88",
         ["66", "76", "100", "120"]),
        ("How many degrees are in a right angle?", "90",
         ["45", "60", "120", "180"]),
        ("How many degrees are in a circle?", "360",
         ["180", "270", "300", "400"]),
        ("How many feet are in a yard?", "3", ["2", "4", "6", "12"]),
        ("How many inches are in a foot?", "12", ["10", "11", "14", "16"]),
        ("How many millimetres are in a centimetre?", "10",
         ["5", "8", "12", "100"]),
        ("How many centimetres are in a metre?", "100",
         ["10", "50", "1000", "10000"]),
        ("How many grams are in a kilogram?", "1000",
         ["100", "500", "10000", "1000000"]),
        ("How many sides does an octagon have?", "8", ["6", "7", "9", "10"]),
        ("How many tentacles does an octopus typically have?", "8",
         ["4", "6", "10", "12"]),
        ("How many legs does a horse have?", "4", ["2", "3", "6", "8"]),
        ("How many legs does a chicken have?", "2", ["3", "4", "6", "8"]),
        ("How many strings on a standard violin?", "4", ["3", "5", "6", "7"]),
        ("How many points on a standard star (typical drawing)?", "5",
         ["3", "4", "6", "8"]),
        ("How many notes are in a major scale?", "7", ["5", "6", "8", "12"]),
        ("How many primary colors of light are there?", "3",
         ["2", "4", "5", "6"]),
        ("How many vowels are in the English alphabet?", "5",
         ["3", "4", "6", "7"]),
        ("How many seasons are there in a year?", "4", ["2", "3", "5", "6"]),
        ("How many wheels does a unicycle have?", "1", ["2", "3", "4", "6"]),
        ("How many wheels does a motorcycle have?", "2", ["1", "3", "4", "6"]),
        ("How many quarts are in a gallon?", "4", ["2", "3", "5", "8"]),
        ("How many cups are in a pint?", "2", ["1", "3", "4", "8"]),
        ("How many pints are in a quart?", "2", ["1", "3", "4", "8"]),
        ("How many holes does a typical golf course have?", "18",
         ["9", "12", "16", "20"]),
        ("How many players are on a soccer team on the field?", "11",
         ["9", "10", "12", "15"]),
        ("How many strikes before you're out in baseball?", "3",
         ["2", "4", "5", "6"]),
    ]
    out: list[dict] = []
    for q, ans, distractors in facts:
        d = rng.sample(distractors, 3)
        out.append(make_4_choice(q, ans, d, rng))
    return out


def gen_animal_classes(rng: random.Random) -> list[dict]:
    out: list[dict] = []
    classes = ["mammal", "bird", "fish", "reptile", "amphibian", "insect"]
    for animal, cls in ANIMALS:
        distractors = rng.sample([c for c in classes if c != cls], 3)
        out.append(make_4_choice(
            f"Which class of animal is {animal}?", cls, distractors, rng,
        ))
    return out


def gen_chemistry(rng: random.Random) -> list[dict]:
    out: list[dict] = []
    pool = [n for _, n in CHEMISTRY]
    for formula, name in CHEMISTRY:
        distractors = rng.sample([n for n in pool if n != name], 3)
        out.append(make_4_choice(
            f"What is the common name for the substance with formula {formula}?",
            name, distractors, rng,
        ))
    return out


def gen_synonyms(rng: random.Random) -> list[dict]:
    out: list[dict] = []
    for word, options, correct_idx in SYNONYM_PAIRS:
        opts = list(options)
        correct = opts[correct_idx]
        distractors = [o for i, o in enumerate(opts) if i != correct_idx]
        out.append(make_4_choice(
            f"Which word is closest in meaning to '{word}'?",
            correct, distractors, rng,
        ))
    return out


# Winogrande-style 2-choice with proper-name fillers. The 'context' is a
# sentence with a blank that requires picking which of two referents fits.
WINO_TEMPLATES = [
    ("Anna is much taller than Beth, so _ can reach the top shelf.", "Anna", "Beth"),
    ("Anna is much taller than Beth, so _ cannot reach the top shelf.", "Beth", "Anna"),
    ("Mark studied hard while Tom watched TV, so _ did better on the test.", "Mark", "Tom"),
    ("Mark studied hard while Tom watched TV, so _ did worse on the test.", "Tom", "Mark"),
    ("Sara is older than Lisa, so _ has been alive longer.", "Sara", "Lisa"),
    ("Sara is older than Lisa, so _ has been alive a shorter time.", "Lisa", "Sara"),
    ("The cat is faster than the dog, so _ catches more mice.", "the cat", "the dog"),
    ("The cat is faster than the dog, so _ catches fewer mice.", "the dog", "the cat"),
    ("John is a better cook than Steve, so _ usually makes dinner.", "John", "Steve"),
    ("John is a better cook than Steve, so _ usually washes dishes.", "Steve", "John"),
    ("The summer was hotter than the spring, so _ felt uncomfortable outside.", "the summer", "the spring"),
    ("The summer was hotter than the spring, so _ felt comfortable outside.", "the spring", "the summer"),
    ("Mary speaks louder than Jane, so _ is easier to hear from far away.", "Mary", "Jane"),
    ("Mary speaks louder than Jane, so _ is harder to hear from far away.", "Jane", "Mary"),
    ("The truck is bigger than the car, so _ can carry more cargo.", "the truck", "the car"),
    ("The truck is bigger than the car, so _ can carry less cargo.", "the car", "the truck"),
    ("Peter is more careful than Mike, so _ rarely makes mistakes.", "Peter", "Mike"),
    ("Peter is more careful than Mike, so _ often makes mistakes.", "Mike", "Peter"),
    ("Lucy can swim better than Emma, so _ is more likely to win the race.", "Lucy", "Emma"),
    ("Lucy can swim better than Emma, so _ is less likely to win the race.", "Emma", "Lucy"),
    ("The lion is stronger than the deer, so _ is the predator.", "the lion", "the deer"),
    ("The lion is stronger than the deer, so _ is the prey.", "the deer", "the lion"),
    ("Owen is a generous tipper while Ralph is stingy, so _ leaves more money on the table.",
     "Owen", "Ralph"),
    ("Owen is a generous tipper while Ralph is stingy, so _ leaves less money on the table.",
     "Ralph", "Owen"),
    ("The professor explained the topic clearly while the student rambled, so _ was easier to follow.",
     "the professor", "the student"),
    ("The professor explained the topic clearly while the student rambled, so _ was harder to follow.",
     "the student", "the professor"),
    ("Kelly arrived on time but Greg arrived late, so _ kept the meeting waiting.",
     "Greg", "Kelly"),
    ("Kelly arrived on time but Greg arrived late, so _ did not keep the meeting waiting.",
     "Kelly", "Greg"),
    ("The new phone is more expensive than the old one, so _ costs more.", "the new phone", "the old one"),
    ("The new phone is more expensive than the old one, so _ costs less.", "the old one", "the new phone"),
    ("Helen is a vegetarian while Daniel eats meat, so _ ordered the salad.",
     "Helen", "Daniel"),
    ("Helen is a vegetarian while Daniel eats meat, so _ ordered the steak.",
     "Daniel", "Helen"),
    ("The mountain is higher than the hill, so _ is harder to climb.", "the mountain", "the hill"),
    ("The mountain is higher than the hill, so _ is easier to climb.", "the hill", "the mountain"),
    ("Robin runs faster than Casey, so _ usually wins their races.", "Robin", "Casey"),
    ("Robin runs faster than Casey, so _ usually loses their races.", "Casey", "Robin"),
    ("The diamond is harder than the gold, so _ scratches the other.", "the diamond", "the gold"),
    ("The diamond is harder than the gold, so _ gets scratched by the other.", "the gold", "the diamond"),
    ("Ahmed is more punctual than Brian, so _ is rarely late.", "Ahmed", "Brian"),
    ("Ahmed is more punctual than Brian, so _ is often late.", "Brian", "Ahmed"),
    ("The river is shallower than the lake, so _ is safer to wade in.", "the river", "the lake"),
    ("The river is shallower than the lake, so _ is more dangerous to wade in.", "the lake", "the river"),
    ("Julia practiced piano daily, while Alex barely practiced, so _ became more skilled.",
     "Julia", "Alex"),
    ("Julia practiced piano daily, while Alex barely practiced, so _ became less skilled.",
     "Alex", "Julia"),
    ("The cake is sweeter than the bread, so _ tastes more like dessert.", "the cake", "the bread"),
    ("The cake is sweeter than the bread, so _ tastes less like dessert.", "the bread", "the cake"),
    ("Connor is taller than Jordan, so _ has to duck through low doorways.", "Connor", "Jordan"),
    ("Connor is taller than Jordan, so _ does not have to duck through low doorways.", "Jordan", "Connor"),
    ("The microwave heats food faster than the oven, so _ is quicker for reheating leftovers.",
     "the microwave", "the oven"),
    ("The microwave heats food faster than the oven, so _ takes longer for reheating leftovers.",
     "the oven", "the microwave"),
    ("Anita is more cautious than Beatrice, so _ rarely takes risks.", "Anita", "Beatrice"),
    ("Anita is more cautious than Beatrice, so _ often takes risks.", "Beatrice", "Anita"),
]


def gen_winogrande(rng: random.Random, n: int = 300) -> list[dict]:
    """Sample with replacement to reach `n`. Each sample re-shuffles letter."""
    out: list[dict] = []
    for _ in range(n):
        ctx, correct, wrong = rng.choice(WINO_TEMPLATES)
        out.append(make_2_choice(ctx, correct, wrong, rng))
    return out


def main() -> int:
    rng = random.Random(SEED)
    examples: list[dict] = []
    examples += gen_capitals(rng)
    examples += gen_colors(rng)
    examples += gen_arithmetic(rng, n=200)
    examples += gen_counts(rng)
    examples += gen_animal_classes(rng)
    examples += gen_chemistry(rng)
    examples += gen_synonyms(rng)
    examples += gen_winogrande(rng, n=300)

    rng.shuffle(examples)

    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    with OUT_PATH.open("w", encoding="utf-8") as fp:
        for ex in examples:
            fp.write(json.dumps(ex, ensure_ascii=False) + "\n")

    print(f"Wrote {len(examples)} examples to {OUT_PATH}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
