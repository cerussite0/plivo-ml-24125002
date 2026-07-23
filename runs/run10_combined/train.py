"""Run 9 trainer: WSD (Warmup-Stable-Decay) learning rate schedule.
Replaces cosine decay with three phases:
  - Warmup: linear ramp (steps 1-100)
  - Stable: constant peak LR (steps 101-1600)
  - Decay: linear decay to min_lr (steps 1601-2000, 20% of total)
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

def get_lr_wsd(step, warmup, total, peak, minimum, decay_frac=0.2):
    """Warmup-Stable-Decay schedule.
    decay_frac: fraction of total steps spent in decay phase (default 20%).
    """
    if step < warmup:
        # Warmup phase: linear ramp
        return peak * step / warmup
    decay_start = int(total * (1 - decay_frac))
    if step < decay_start:
        # Stable phase: constant peak LR
        return peak
    # Decay phase: linear decay from peak to minimum
    ratio = (step - decay_start) / max(1, total - decay_start)
    return peak + (minimum - peak) * ratio

def get_lr_cosine(step, warmup, total, peak, minimum):
    """Original cosine schedule for comparison."""
    if step < warmup:
        return peak * step / warmup
    ratio = (step - warmup) / max(1, total - warmup)
    return minimum + (peak - minimum) * 0.5 * (1 + math.cos(math.pi * ratio))

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True)
    ap.add_argument("--steps", type=int, default=2000)
    ap.add_argument("--batch", type=int, default=8)
    ap.add_argument("--accum", type=int, default=4)  # Use reduced from Run 8
    ap.add_argument("--dropout", type=float, default=0.05)  # From Run 8
    ap.add_argument("--schedule", choices=["wsd", "cosine"], default="wsd")
    ap.add_argument("--decay_frac", type=float, default=0.2)  # 20% decay phase
    ap.add_argument("--use_muon", action="store_true")
    ap.add_argument("--muon_lr", type=float, default=0.02)
    ap.add_argument("--adam_lr", type=float, default=6e-4)
    ap.add_argument("--min_lr_ratio", type=float, default=0.1)
    ap.add_argument("--warmup", type=int, default=100)
    ap.add_argument("--wd", type=float, default=0.1)
    ap.add_argument("--clip", type=float, default=1.0)
    ap.add_argument("--seed", type=int, default=1337)
    ap.add_argument("--out", default="ckpt.pt")
    ap.add_argument("--log_every", type=int, default=100)
    args = ap.parse_args()
    assert args.steps <= MAX_STEPS
    torch.manual_seed(args.seed)
    device = "cpu"
    print(f"device: {device}, schedule: {args.schedule}, decay_frac: {args.decay_frac}")

    # Load pre-encoded token IDs
    import os
    runs_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    ids_path = os.path.join(runs_dir, "train_ids_v1024.pt")
    if os.path.exists(ids_path):
        ids = torch.load(ids_path, weights_only=True)
        print(f"loaded pre-encoded ids: {len(ids):,} tokens")
    else:
        text = open(args.data, encoding="utf-8").read()
        tok = tokenizer_mod.load()
        ids = torch.tensor(tok.encode(text), dtype=torch.long)
        print(f"encoded corpus: {len(ids):,} tokens")
    tok = tokenizer_mod.load()
    print(f"vocab {tok.vocab_size}, {len(ids):,} tokens")

    cfg = Config()
    cfg.vocab_size = tok.vocab_size
    cfg.dropout = args.dropout
    model = GPT(cfg).to(device)
    n = model.n_params()
    print(f"model: {n:,} params  (budget: {MAX_PARAMS:,})")
    assert n <= MAX_PARAMS

    if args.use_muon:
        muon_params = []
        adam_decay = []
        adam_nodecay = []
        for name, p in model.named_parameters():
            if not p.requires_grad:
                continue
            if p.ndim == 2 and 'tok_emb' not in name:
                muon_params.append(p)
            elif p.ndim >= 2:
                adam_decay.append(p)
            else:
                adam_nodecay.append(p)
        opt_muon = Muon(muon_params, lr=args.muon_lr, momentum=0.95, ns_steps=5)
        opt_adam = torch.optim.AdamW([
            {"params": adam_decay, "weight_decay": args.wd},
            {"params": adam_nodecay, "weight_decay": 0.0},
        ], lr=args.adam_lr, betas=(0.9, 0.95))
        optimizers = [opt_muon, opt_adam]
    else:
        decay_p = [p for p in model.parameters() if p.dim() >= 2]
        nodecay_p = [p for p in model.parameters() if p.dim() < 2]
        opt_adam = torch.optim.AdamW([
            {"params": decay_p, "weight_decay": args.wd},
            {"params": nodecay_p, "weight_decay": 0.0},
        ], lr=args.adam_lr, betas=(0.9, 0.95))
        optimizers = [opt_adam]

    # Select schedule function
    if args.schedule == "wsd":
        get_lr = lambda step, peak, minimum: get_lr_wsd(
            step, args.warmup, args.steps, peak, minimum, args.decay_frac)
    else:
        get_lr = lambda step, peak, minimum: get_lr_cosine(
            step, args.warmup, args.steps, peak, minimum)

    model.train()
    t0 = time.time()
    losses = []
    for step in range(1, args.steps + 1):
        muon_lr = get_lr(step, args.muon_lr, args.muon_lr * args.min_lr_ratio)
        adam_lr = get_lr(step, args.adam_lr, args.adam_lr * args.min_lr_ratio)

        for opt in optimizers:
            for pg in opt.param_groups:
                if isinstance(opt, Muon):
                    pg["lr"] = muon_lr
                else:
                    pg["lr"] = adam_lr

        for opt in optimizers:
            opt.zero_grad(set_to_none=True)

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

        for opt in optimizers:
            opt.step()

        losses.append(accum_loss)
        if step % args.log_every == 0 or step == 1:
            avg = sum(losses[-args.log_every:]) / len(losses[-args.log_every:])
            ms = (time.time() - t0) / step * 1000
            print(f"step {step:5d}  loss {avg:.4f}  lr {adam_lr:.2e}  ({ms:.0f} ms/step)")

    torch.save({"model": model.state_dict(),
                "config": {k: getattr(cfg, k) for k in dir(cfg)
                           if not k.startswith("_") and not callable(getattr(cfg, k))},
                "steps": args.steps, "train_loss_curve": losses}, args.out)
    print(f"saved {args.out}  ({time.time()-t0:.0f}s total)")

if __name__ == "__main__":
    main()
