# Data
## Preprocessing
- EDA really explore what is in the data and write a list of what needs to be cleaned
- Extend the notebook exploration to explicitly tag HTML/code blocks, strange headers/formatting artifacts, and other noisy samples so the cleaning requirements are backed by concrete examples.
- Implement the list from EDA (now only non-english removal and the eval dataset removel is in there)
- Add a cleaning stage in `src/data/preprocess.py` to strip HTML tags, drop code snippets/markdown dumps, normalize or remove corrupted characters, and filter out noisy samples based on the EDA findings before tokenization.
- Wire new CLI args (e.g. `--num-train-samples` and `--seed`) through `main.py` → `run_preprocess` so the preprocessing can reproducibly operate on a configurable subset size instead of the current small/full toggle.
## Sample selection
- Choose a smart way to select the training data when parameter is smaller than the trainingset
- define and implement a way to select the best possible data 
- Once the CLI args exist, implement deterministic subsampling logic (respecting the seed) so we consistently pick the same subset of OpenWebText for experiments.
