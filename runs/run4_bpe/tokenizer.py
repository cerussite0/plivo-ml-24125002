"""BPE tokenizer with byte-level fallback. Lossless: decode(encode(text)) == text.
Uses numpy-accelerated encoding for large texts."""
import json
import os
import numpy as np


class BPETokenizer:
    def __init__(self, merges, vocab_size):
        self.merges = [tuple(m) for m in merges]
        self.vocab_size = vocab_size
        # Build decode table: token_id -> bytes
        self._decode_table = {i: bytes([i]) for i in range(256)}
        for i, (a, b) in enumerate(self.merges):
            self._decode_table[256 + i] = self._decode_table[a] + self._decode_table[b]

    def encode(self, text):
        """Encode text to token ids using numpy for speed."""
        raw = text.encode("utf-8")
        data = np.array(list(raw), dtype=np.int32)
        for rank, (a, b) in enumerate(self.merges):
            if len(data) < 2:
                break
            new_id = 256 + rank
            # Find positions where pair (a,b) occurs
            left = data[:-1]
            right = data[1:]
            mask = (left == a) & (right == b)
            if not mask.any():
                continue
            positions = np.where(mask)[0]
            # Build new array: mark positions to skip
            skip = np.zeros(len(data), dtype=bool)
            skip[positions + 1] = True  # skip the 'b' of each pair
            # Handle overlapping merges: if position i and i+1 both want to merge,
            # only merge at i (greedy left-to-right)
            for p in positions:
                if p > 0 and skip[p]:  # this 'a' was already consumed as 'b' of prev merge
                    skip[p + 1] = False  # undo the skip of this pair's 'b'
            # Replace 'a' with new_id at merge positions (that weren't skipped)
            result = []
            i = 0
            while i < len(data):
                if i < len(data) - 1 and data[i] == a and data[i+1] == b and not skip[i]:
                    result.append(new_id)
                    i += 2
                elif skip[i]:
                    i += 1
                else:
                    result.append(int(data[i]))
                    i += 1
            data = np.array(result, dtype=np.int32)
        return data.tolist()

    def decode(self, ids):
        """Decode token ids to text. Lossless for valid input."""
        raw = b"".join(self._decode_table[i] for i in ids)
        return raw.decode("utf-8", errors="replace")

    def save(self, path):
        with open(path, "w") as f:
            json.dump({"type": "bpe", "merges": self.merges,
                       "vocab_size": self.vocab_size}, f)

    @classmethod
    def from_file(cls, path):
        with open(path) as f:
            data = json.load(f)
        return cls(data["merges"], data["vocab_size"])


def load(path=None):
    """Return the tokenizer. Looks for bpe_merges.json next to this file."""
    merges_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "bpe_merges.json")
    if os.path.exists(merges_path):
        return BPETokenizer.from_file(merges_path)
    raise FileNotFoundError(f"No bpe_merges.json found at {merges_path}")
