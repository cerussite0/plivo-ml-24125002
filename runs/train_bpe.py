"""Ultra-fast BPE trainer using numpy for pair counting."""
import json, sys, time, os
import numpy as np
sys.path.insert(0, os.path.dirname(__file__))

def train_bpe_numpy(raw_bytes, n_merges, verbose=True):
    """Train BPE using numpy vectorized pair counting."""
    data = np.array(list(raw_bytes), dtype=np.int32)
    merges = []
    t0 = time.time()
    
    for mi in range(n_merges):
        if len(data) < 2:
            break
        # Vectorized pair counting: create pair keys and count with bincount
        left = data[:-1]
        right = data[1:]
        # Encode pairs as single int: left * max_vocab + right
        max_v = 256 + mi  # current max token id
        pair_keys = left.astype(np.int64) * (max_v + 1) + right.astype(np.int64)
        counts = np.bincount(pair_keys)
        best_key = counts.argmax()
        best_count = counts[best_key]
        if best_count < 2:
            break
        a = int(best_key // (max_v + 1))
        b = int(best_key % (max_v + 1))
        new_id = 256 + mi
        merges.append((a, b))
        
        # Apply merge: find all positions where pair (a,b) occurs
        mask = (left == a) & (right == b)
        positions = np.where(mask)[0]
        if len(positions) == 0:
            break
        
        # Build new sequence
        new_data = []
        i = 0
        pos_set = set(positions.tolist())
        for i in range(len(data)):
            if i in pos_set:
                new_data.append(new_id)
            elif (i - 1) in pos_set:
                continue  # skip the 'b' of a merged pair
            else:
                new_data.append(data[i])
        data = np.array(new_data, dtype=np.int32)
        
        if verbose and ((mi+1) % 50 == 0 or mi+1 == n_merges or mi == 0):
            print(f"  merge {mi+1}/{n_merges}  pair=({a},{b})  count={best_count:,}  "
                  f"seq_len={len(data):,}  ({time.time()-t0:.1f}s)")
    
    return merges

def main():
    corpus_path = sys.argv[1]
    n_merges = int(sys.argv[2]) if len(sys.argv) > 2 else 768
    out_path = sys.argv[3] if len(sys.argv) > 3 else "bpe_merges.json"
    sample_kb = int(sys.argv[4]) if len(sys.argv) > 4 else 200

    raw = open(corpus_path, "rb").read()
    sample_bytes = sample_kb * 1024
    if len(raw) > sample_bytes:
        print(f"Subsampling: {len(raw):,} -> {sample_bytes:,} bytes")
        raw = raw[:sample_bytes]

    print(f"Training BPE: {n_merges} merges on {len(raw):,} bytes")
    merges = train_bpe_numpy(raw, n_merges)

    with open(out_path, "w") as f:
        json.dump({"merges": merges, "vocab_size": 256 + len(merges)}, f)
    print(f"\nSaved {len(merges)} merges to {out_path} (vocab={256+len(merges)})")

    if len(merges) >= 512:
        sub = out_path.replace(".json", "_512.json")
        with open(sub, "w") as f:
            json.dump({"merges": merges[:512], "vocab_size": 768}, f)
        print(f"Saved 512-merge subset to {sub}")

    # Round-trip test
    from tokenizer_bpe import BPETokenizer
    tok = BPETokenizer(merges, 256 + len(merges))
    for test in ["Hello world!", "यह एक परीक्षा है।", "abc 123 !@#"]:
        enc = tok.encode(test)
        dec = tok.decode(enc)
        byt = len(test.encode("utf-8"))
        print(f"  '{test}': {byt}B -> {len(enc)} tok ({byt/max(len(enc),1):.1f}x)")
        assert dec == test, f"FAIL: {dec!r} != {test!r}"
    print("All round-trip tests OK ✓")

if __name__ == "__main__":
    main()
