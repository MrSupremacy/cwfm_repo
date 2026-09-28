"""Small real-asset F0 regression; NO formal train/capture/metric writes.

Only two validation examples/task, one disposable R4-hard optimizer update,
and a temporary checkpoint. Reports are diagnostic, not experiment results.
"""
from pathlib import Path
from tempfile import TemporaryDirectory
import argparse
import hashlib
import sys

ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(ROOT / "src"), str(ROOT)]


def main():
    import torch
    from task5.common.config import Condition, load_config, protocol_id, root_for
    from task5.common.context import load_context
    from task5.common.io import write_json
    from task5.common.randomness import configure_torch, seed_dropout
    from task5.data.datasets import make_loader, move_batch
    from task5.training.checkpoints import restore_checkpoint, save_checkpoint
    from task5.aggregation.phase_a import check_base

    p = argparse.ArgumentParser()
    p.add_argument("--local", default=str(ROOT / "configs/extensions/phaseA_f0.yaml"))
    args = p.parse_args()
    config = load_config(ROOT / "configs/suites/phaseA_f0.yaml", args.local)
    configure_torch(config)
    run = "formal20260830a"
    check_base(config, run)
    observations = []
    for task in config["suite"]["tasks"]:
        for arm in ("R2-soft", "R4-hard"):
            c = Condition(task, arm, "default", 6, 0 if arm == "R4-hard" else None)
            model, tokenizer, ctrl, data, header, _ = load_context(config, c, run)
            batch = next(iter(make_loader(config, data.select([0, 1]), tokenizer, 2)))
            x = move_batch(batch, config["execution"]["device"])
            seen = {}
            def observe(w, shape, valid, selected, q, activation):
                assert q is not None and q.shape[1] == 64
                s = selected[valid]
                counts = torch.bincount(s.flatten(), minlength=64)
                assert counts.sum() == valid.sum() * 6
                seen[w.key] = int(valid.sum())
            model.eval()
            ctrl.teacher_batch(x)
            ctrl.observe(observe, with_q=True)
            with torch.no_grad():
                logits = model(**x, use_cache=False).logits.detach()
            assert len(seen) == 12
            ctrl.generation()
            with torch.no_grad():
                prediction = model.generate(input_ids=x["input_ids"], attention_mask=x["attention_mask"],
                                            max_new_tokens=config["capture"]["max_new_tokens"], do_sample=False, num_beams=1)
            trainable = [v for v in model.parameters() if v.requires_grad]
            record = {"task": task, "arm": arm, "valid_tokens": seen, "samples": 2,
                      "trainable_parameters": sum(p.numel() for p in trainable),
                      "predictions": tokenizer.batch_decode(prediction, skip_special_tokens=True)}
            if arm == "R2-soft":
                assert not trainable
            else:
                assert len(trainable) == 12
                frozen = [(name, p) for name, p in model.named_parameters() if not p.requires_grad]
                def frozen_hash():
                    h = hashlib.sha256()
                    for name, value in frozen:
                        h.update(name.encode())
                        h.update(value.detach().cpu().numpy().tobytes())
                    return h.hexdigest()
                before = frozen_hash()
                initial = {key: w.router.summary.detach().clone() for key, w in ctrl.wrappers.items()}
                optimizer = torch.optim.Adam(trainable, lr=config["training"]["lr"], foreach=False, fused=False)
                scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lambda _: 1.)
                shuffle = torch.Generator().manual_seed(0)
                seed_dropout(0)
                model.train()
                ctrl.teacher_batch(x)
                loss = model(**x, use_cache=False).loss
                loss.backward()
                gradients = {key: w.router.summary.grad.abs().sum().item() for key, w in ctrl.wrappers.items()}
                assert all(value > 0 for value in gradients.values())
                assert all(p.grad is None for _, p in frozen)
                torch.nn.utils.clip_grad_norm_(trainable, 1., error_if_nonfinite=True)
                optimizer.step()
                scheduler.step()
                assert frozen_hash() == before
                assert all(not torch.equal(w.router.summary, initial[key]) for key, w in ctrl.wrappers.items())
                model.eval()
                ctrl.teacher_batch(x)
                with torch.no_grad():
                    expected = model(**x, use_cache=False).logits.detach()
                with TemporaryDirectory(prefix="task5_phaseA_f0_test_") as temp:
                    path = Path(temp) / "checkpoint"
                    save_checkpoint(path, ctrl, optimizer, scheduler, shuffle, {"name": "test", "step": 1, "epoch": 0})
                    with torch.no_grad():
                        for value in trainable:
                            value.add_(1)
                    restore_checkpoint(path, ctrl)
                    with torch.no_grad():
                        actual = model(**x, use_cache=False).logits
                    assert torch.equal(expected, actual)
                record.update(loss=float(loss.detach()), gradients=gradients, frozen_sha256=before, exact_reload=True)
            observations.append(record)
            print(f"PASS {task}/{arm}: actual input assets, two-sample forward, F0 parameter scope")
            del model, ctrl, logits, trainable
            torch.cuda.empty_cache()
    write_json(root_for(config) / "tmp/phaseA_f0-tests/real_assets.json",
               {"scope": "diagnostic only; NOT formal experiment results", "protocol": protocol_id(config),
                "observations": observations, "success": True})


if __name__ == "__main__":
    main()
