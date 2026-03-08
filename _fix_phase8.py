from pathlib import Path

p = Path("src/data/preprocess.py")
c = p.read_text()

old = (
    "    eos_id = tokenizer.eos_token_id if tokenizer.eos_token_id is not None else 0\n"
    "\n"
    "    def _pad_to_chunk(arr: np.ndarray) -> np.ndarray:\n"
    "        remainder = len(arr) % chunk_size\n"
    "        if remainder == 0:\n"
    "            return arr\n"
    "        return np.concatenate([arr, np.full(chunk_size - remainder, eos_id, dtype=arr.dtype)])"
)

new = (
    "    def _pad_to_chunk(arr: np.ndarray) -> np.ndarray:\n"
    "        remainder = len(arr) % chunk_size\n"
    "        if remainder == 0:\n"
    "            return arr\n"
    "        return arr[:-remainder]  # trim tail, never pad with EOS"
)

if old in c:
    p.write_text(c.replace(old, new, 1))
    print("REPLACED OK")
else:
    print("NO MATCH — dumping context:")
    idx = c.find("Pad both splits")
    print(repr(c[idx:idx+600]))
