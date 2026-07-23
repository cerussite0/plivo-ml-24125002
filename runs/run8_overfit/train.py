"""Run 8 trainer: Overfitting mitigation.
Same architecture as Run 5, but:
- Reduced accum 8->4 (epochs 24.2 -> 12.1)
- Added dropout=0.05
- Uses best optimizer from Run 7 (Muon or AdamW depending on results)
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
    ap.add_argument("--accum", type=int, default=4)  # Reduced from 8!
    ap.add_argument("--dropout", type=float, default=0.05)  # Light regularization
    ap.add_argument("--use_muon", action="store_true")  # Toggle optimizer
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
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"device: {device}")

    # Load pre-encoded token IDs (much faster than re-encoding 7MB)
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
    tokens_per_step = args.batch * args.accum * 512
    epochs = (tokens_per_step * args.steps) / len(ids)
    print(f"vocab {tok.vocab_size}, effective batch: {args.batch * args.accum}, pseudo-epochs: {epochs:.1f}")

    cfg = Config()
    cfg.vocab_size = tok.vocab_size
    cfg.dropout = args.dropout  # Enable dropout
    model = GPT(cfg).to(device)
    n = model.n_params()
    print(f"model: {n:,} params  (budget: {MAX_PARAMS:,})")
    assert n <= MAX_PARAMS

    if args.use_muon:
        # Muon for 2D weights, AdamW for rest
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
        print(f"Using Muon+AdamW (muon_lr={args.muon_lr}, adam_lr={args.adam_lr})")
    else:
        decay_p = [p for p in model.parameters() if p.dim() >= 2]
        nodecay_p = [p for p in model.parameters() if p.dim() < 2]
        opt_adam = torch.optim.AdamW([
            {"params": decay_p, "weight_decay": args.wd},
            {"params": nodecay_p, "weight_decay": 0.0},
        ], lr=args.adam_lr, betas=(0.9, 0.95))
        optimizers = [opt_adam]
        print(f"Using AdamW (lr={args.adam_lr})")

    model.train()
    t0 = time.time()
    losses = []
    for step in range(1, args.steps + 1):
        muon_lr = get_lr(step, args.warmup, args.steps,
                         args.muon_lr, args.muon_lr * args.min_lr_ratio)
        adam_lr = get_lr(step, args.warmup, args.steps,
                         args.adam_lr, args.adam_lr * args.min_lr_ratio)

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
