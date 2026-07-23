"""Shared trainer for BPE runs (4-6). AdamW + cosine LR + grad accum + label smoothing."""
import argparse, math, time, torch, torch.nn.functional as F
from model import GPT, Config
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
    ap.add_argument("--lr", type=float, default=6e-4)
    ap.add_argument("--min_lr", type=float, default=6e-5)
    ap.add_argument("--warmup", type=int, default=100)
    ap.add_argument("--wd", type=float, default=0.1)
    ap.add_argument("--clip", type=float, default=1.0)
    ap.add_argument("--label_smooth", type=float, default=0.0)
    ap.add_argument("--seed", type=int, default=1337)
    ap.add_argument("--out", default="ckpt.pt")
    ap.add_argument("--log_every", type=int, default=200)
    args = ap.parse_args()
    assert args.steps <= MAX_STEPS
    torch.manual_seed(args.seed)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"device: {device}")

    text = open(args.data, encoding="utf-8").read()
    tok = tokenizer_mod.load()
    ids = torch.tensor(tok.encode(text), dtype=torch.long)
    raw_bytes = len(text.encode("utf-8"))
    print(f"corpus: {raw_bytes:,} bytes -> {len(ids):,} tokens "
          f"(vocab {tok.vocab_size}, {raw_bytes/len(ids):.2f}x compression)")

    cfg = Config()
    cfg.vocab_size = tok.vocab_size
    model = GPT(cfg).to(device)
    n = model.n_params()
    print(f"model: {n:,} params  (budget: {MAX_PARAMS:,})")
    assert n <= MAX_PARAMS, f"over budget: {n} > {MAX_PARAMS}"

    decay_p = [p for p in model.parameters() if p.dim() >= 2]
    nodecay_p = [p for p in model.parameters() if p.dim() < 2]
    opt = torch.optim.AdamW([
        {"params": decay_p, "weight_decay": args.wd},
        {"params": nodecay_p, "weight_decay": 0.0},
    ], lr=args.lr, betas=(0.9, 0.95))

    ls = args.label_smooth
    model.train()
    t0 = time.time()
    losses = []
    for step in range(1, args.steps + 1):
        lr = get_lr(step, args.warmup, args.steps, args.lr, args.min_lr)
        for pg in opt.param_groups:
            pg["lr"] = lr
        opt.zero_grad(set_to_none=True)
        accum_loss = 0.0
        for _ in range(args.accum):
            x, y = get_batch(ids, cfg.block_size, args.batch, device)
            logits, _ = model(x, None)
            loss = F.cross_entropy(logits.view(-1, logits.size(-1)), y.reshape(-1),
                                   label_smoothing=ls)
            loss = loss / args.accum
            loss.backward()
            accum_loss += loss.item()
        if args.clip > 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), args.clip)
        opt.step()
        losses.append(accum_loss)
        if step % args.log_every == 0 or step == 1:
            avg = sum(losses[-args.log_every:]) / len(losses[-args.log_every:])
            ms = (time.time() - t0) / step * 1000
            print(f"step {step:5d}  loss {avg:.4f}  lr {lr:.2e}  ({ms:.0f} ms/step)")

    torch.save({"model": model.state_dict(),
                "config": {k: getattr(cfg, k) for k in dir(cfg)
                           if not k.startswith("_") and not callable(getattr(cfg, k))},
                "steps": args.steps, "train_loss_curve": losses}, args.out)
    print(f"saved {args.out}  ({time.time()-t0:.0f}s total)")

if __name__ == "__main__":
    main()
