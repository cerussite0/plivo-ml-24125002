"""Run 7 trainer: Muon optimizer for 2D weights + AdamW for rest.
Muon uses Newton-Schulz orthogonalized momentum for projection matrices.
AdamW handles embeddings, norms, and biases.
"""
import argparse, math, time, torch, torch.nn.functional as F
from model import GPT, Config
from muon import Muon
import tokenizer as tokenizer_mod

MAX_STEPS = 2000
MAX_PARAMS = 2_000_000

def get_batch(ids, block, batch, device):
    ix = torch.randint(len(ids) - block - 1, (batch,))
    x = torch.stack([ids[i:i + block] for i in ix])
    y = torch.stack([ids[i + 1:i + 1 + block] for i in ix])
    return x.to(device), y.to(device)

def get_lr(step, warmup, total, peak, minimum):
    if step < warmup:
        return peak * step / warmup
    ratio = (step - warmup) / max(1, total - warmup)
    return minimum + (peak - minimum) * 0.5 * (1 + math.cos(math.pi * ratio))

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True)
    ap.add_argument("--steps", type=int, default=2000)
    ap.add_argument("--batch", type=int, default=8)
    ap.add_argument("--accum", type=int, default=8)
    ap.add_argument("--muon_lr", type=float, default=0.02)
    ap.add_argument("--adam_lr", type=float, default=6e-4)
    ap.add_argument("--min_lr_ratio", type=float, default=0.1)
    ap.add_argument("--warmup", type=int, default=100)
    ap.add_argument("--wd", type=float, default=0.1)
    ap.add_argument("--muon_wd", type=float, default=0.0)
    ap.add_argument("--clip", type=float, default=1.0)
    ap.add_argument("--seed", type=int, default=1337)
    ap.add_argument("--out", default="ckpt.pt")
    ap.add_argument("--log_every", type=int, default=200)
    args = ap.parse_args()
    assert args.steps <= MAX_STEPS
    torch.manual_seed(args.seed)
    device = "cpu"
    print(f"device: {device}")

    # Load pre-encoded token IDs (much faster than re-encoding 7MB)
    import os
    runs_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    ids_path = os.path.join(runs_dir, "train_ids_v1024.pt")
    if os.path.exists(ids_path):
        ids = torch.load(ids_path, weights_only=True)
        print(f"loaded pre-encoded ids from {ids_path}: {len(ids):,} tokens")
    else:
        text = open(args.data, encoding="utf-8").read()
        tok = tokenizer_mod.load()
        ids = torch.tensor(tok.encode(text), dtype=torch.long)
        print(f"encoded corpus: {len(ids):,} tokens")
    tok = tokenizer_mod.load()
    print(f"vocab {tok.vocab_size}, {len(ids):,} tokens")

    cfg = Config()
    cfg.vocab_size = tok.vocab_size
    model = GPT(cfg).to(device)
    n = model.n_params()
    print(f"model: {n:,} params  (budget: {MAX_PARAMS:,})")
    assert n <= MAX_PARAMS, f"over budget: {n} > {MAX_PARAMS}"

    # Split params: 2D weights -> Muon, rest -> AdamW
    muon_params = []
    adam_decay = []
    adam_nodecay = []
    for name, p in model.named_parameters():
        if not p.requires_grad:
            continue
        if p.ndim == 2 and 'tok_emb' not in name:
            # 2D projection weights -> Muon
            muon_params.append(p)
        elif p.ndim >= 2:
            # Embeddings -> AdamW with decay
            adam_decay.append(p)
        else:
            # 1D: biases, RMSNorm weights -> AdamW no decay
            adam_nodecay.append(p)

    print(f"Muon params: {sum(p.numel() for p in muon_params):,} "
          f"({len(muon_params)} tensors)")
    print(f"AdamW decay params: {sum(p.numel() for p in adam_decay):,} "
          f"({len(adam_decay)} tensors)")
    print(f"AdamW no-decay params: {sum(p.numel() for p in adam_nodecay):,} "
          f"({len(adam_nodecay)} tensors)")

    opt_muon = Muon(muon_params, lr=args.muon_lr, momentum=0.95,
                    ns_steps=5, weight_decay=args.muon_wd)
    opt_adam = torch.optim.AdamW([
        {"params": adam_decay, "weight_decay": args.wd},
        {"params": adam_nodecay, "weight_decay": 0.0},
    ], lr=args.adam_lr, betas=(0.9, 0.95))

    model.train()
    t0 = time.time()
    losses = []
    for step in range(1, args.steps + 1):
        # Cosine schedule for both optimizers
        muon_lr = get_lr(step, args.warmup, args.steps,
                         args.muon_lr, args.muon_lr * args.min_lr_ratio)
        adam_lr = get_lr(step, args.warmup, args.steps,
                         args.adam_lr, args.adam_lr * args.min_lr_ratio)
        for pg in opt_muon.param_groups:
            pg["lr"] = muon_lr
        for pg in opt_adam.param_groups:
            pg["lr"] = adam_lr

        opt_muon.zero_grad(set_to_none=True)
        opt_adam.zero_grad(set_to_none=True)

        accum_loss = 0.0
        for _ in range(args.accum):
            x, y = get_batch(ids, cfg.block_size, args.batch, device)
            logits, _ = model(x, None)
            loss = F.cross_entropy(logits.view(-1, logits.size(-1)), y.reshape(-1))
            loss = loss / args.accum
            loss.backward()
            accum_loss += loss.item()

        if args.clip > 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), args.clip)

        opt_muon.step()
        opt_adam.step()

        losses.append(accum_loss)
        if step % args.log_every == 0 or step == 1:
            avg = sum(losses[-args.log_every:]) / len(losses[-args.log_every:])
            ms = (time.time() - t0) / step * 1000
            print(f"step {step:5d}  loss {avg:.4f}  muon_lr {muon_lr:.2e}  "
                  f"adam_lr {adam_lr:.2e}  ({ms:.0f} ms/step)")

    torch.save({"model": model.state_dict(),
                "config": {k: getattr(cfg, k) for k in dir(cfg)
                           if not k.startswith("_") and not callable(getattr(cfg, k))},
                "steps": args.steps, "train_loss_curve": losses}, args.out)
    elapsed = time.time() - t0
    print(f"saved {args.out}  ({elapsed:.0f}s total, {elapsed/args.steps*1000:.0f} ms/step)")

if __name__ == "__main__":
    main()
