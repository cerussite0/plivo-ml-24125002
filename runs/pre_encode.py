"""Pre-encode corpus with BPE. Splits text into lines for fast encoding.
Run ONCE. Saves .pt files that all future runs load directly."""
import sys, os, time, json, torch

def encode_line(line_bytes, merges):
    """Encode a single line (short) with BPE merges. Fast for short inputs."""
    ids = list(line_bytes)
    for rank, (a, b) in enumerate(merges):
        new_id = 256 + rank
        new_ids = []
        i = 0
        while i < len(ids):
            if i < len(ids) - 1 and ids[i] == a and ids[i + 1] == b:
                new_ids.append(new_id)
                i += 2
            else:
                new_ids.append(ids[i])
                i += 1
        ids = new_ids
    return ids

def encode_corpus(text_bytes, merges, label=""):
    """Encode full corpus by splitting into lines (fast per-line encoding)."""
    lines = text_bytes.split(b'\n')
    all_ids = []
    t0 = time.time()
    for i, line in enumerate(lines):
        # Encode line + newline (except last line if empty)
        if i < len(lines) - 1:
            ids = encode_line(line + b'\n', merges)
        else:
            if line:
                ids = encode_line(line, merges)
            else:
                continue
        all_ids.extend(ids)
        if (i + 1) % 5000 == 0:
            elapsed = time.time() - t0
            print(f"  [{label}] {i+1}/{len(lines)} lines, {len(all_ids):,} tokens ({elapsed:.0f}s)")
    elapsed = time.time() - t0
    print(f"  [{label}] DONE: {len(lines)} lines -> {len(all_ids):,} tokens ({elapsed:.0f}s)")
    return all_ids

def main():
    runs_dir = os.path.dirname(os.path.abspath(__file__))
    
    # Load merges
    with open(os.path.join(runs_dir, "bpe_merges_512.json")) as f:
        merges_512 = [tuple(m) for m in json.load(f)["merges"]]
    with open(os.path.join(runs_dir, "bpe_merges.json")) as f:
        merges_768 = [tuple(m) for m in json.load(f)["merges"]]
    print(f"Loaded merges: 512 ({len(merges_512)}) and 768 ({len(merges_768)})")

    # Read files as bytes
    train_path = os.path.join(runs_dir, "..", "llm_handout", "data", "train_corpus.txt")
    dev_path = os.path.join(runs_dir, "..", "llm_handout", "data", "dev_eval.txt")
    train_bytes = open(train_path, "rb").read()
    dev_bytes = open(dev_path, "rb").read()
    print(f"Train: {len(train_bytes):,} bytes, Dev: {len(dev_bytes):,} bytes")

    # Encode everything
    for name, merges, vocab in [("v768", merges_512, 768), ("v1024", merges_768, 1024)]:
        print(f"\n=== {name} (vocab {vocab}, {len(merges)} merges) ===")
        
        # Train corpus
        train_ids = encode_corpus(train_bytes, merges, f"train-{name}")
        out = os.path.join(runs_dir, f"train_ids_{name}.pt")
        torch.save(torch.tensor(train_ids, dtype=torch.long), out)
        compression = len(train_bytes) / len(train_ids)
        print(f"  Saved {out} ({compression:.2f}x compression)")
        
        # Dev corpus
        dev_ids = encode_corpus(dev_bytes, merges, f"dev-{name}")
        out = os.path.join(runs_dir, f"dev_ids_{name}.pt")
        torch.save(torch.tensor(dev_ids, dtype=torch.long), out)
        print(f"  Saved {out}")

    # Verify round-trip on a sample
    print("\n=== Round-trip verification ===")
    decode_table = {i: bytes([i]) for i in range(256)}
    for i, (a, b) in enumerate(merges_768):
        decode_table[256 + i] = decode_table[a] + decode_table[b]
    
    sample = train_bytes[:2000]
    sample_ids = encode_line(list(sample), merges_768)
    decoded = b"".join(decode_table[i] for i in sample_ids)
    # Note: line-based encoding means we need to check line by line
    # But for a single chunk it should be exact
    assert decoded == sample, "Round-trip FAILED!"
    print(f"Round-trip OK: {len(sample)} bytes -> {len(sample_ids)} tokens -> {len(decoded)} bytes")
    
    print("\n=== ALL PRE-ENCODING COMPLETE ===")
    for f in ["train_ids_v768.pt", "train_ids_v1024.pt", "dev_ids_v768.pt", "dev_ids_v1024.pt"]:
        path = os.path.join(runs_dir, f)
        t = torch.load(path, weights_only=True)
        print(f"  {f}: {len(t):,} tokens")

if __name__ == "__main__":
    main()
