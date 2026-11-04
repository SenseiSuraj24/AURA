#!/usr/bin/env python3
"""
scripts/smoke_test.py — fast wiring check for the AURA pipeline
================================================================
Catches the class of breakage where a signature or module changes and an
entry point silently stops working (e.g. a stale kwarg, a missing argument,
a merge that leaves a file unparseable).

Default checks need no dataset and run in seconds:
  1. Every tracked .py file parses.
  2. config.py imports on a fresh clone (no logs/calibration_results.json).
  3. Core modules import.
  4. Models, detector, two-pass local training and DC-FLTrust aggregation
     run end-to-end on synthetic tensors.

  --fl  additionally runs one real in-process federation round
        (needs dataset/, saved_models/ and logs/calibration_results.json).

Usage:
    python scripts/smoke_test.py
    python scripts/smoke_test.py --fl
"""

import argparse
import ast
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

_failures = []


def check(name, fn):
    try:
        fn()
        print(f"  [PASS] {name}")
    except Exception as e:
        print(f"  [FAIL] {name}: {type(e).__name__}: {e}")
        _failures.append(name)


def parse_all_sources():
    tracked = subprocess.run(
        ["git", "ls-files", "*.py"], cwd=ROOT, capture_output=True, text=True, check=True
    ).stdout.split()
    bad = []
    for rel in tracked:
        try:
            ast.parse((ROOT / rel).read_text(encoding="utf-8"), filename=rel)
        except SyntaxError as e:
            bad.append(f"{rel}:{e.lineno}")
    if bad:
        raise SyntaxError(", ".join(bad))


def fresh_clone_config_import():
    with tempfile.TemporaryDirectory() as tmp:
        shutil.copy(ROOT / "config.py", tmp)
        (Path(tmp) / "saved_models").mkdir()
        shutil.copy(ROOT / "saved_models" / "attack_class_stats.json", Path(tmp) / "saved_models")
        proc = subprocess.run(
            [sys.executable, "-c", "import config; assert not config.CH2_SPLIT_CALIBRATED"],
            cwd=tmp, capture_output=True, text=True, encoding="utf-8", errors="replace",
        )
        if proc.returncode != 0:
            raise RuntimeError(proc.stderr.strip().splitlines()[-1])


def import_core_modules():
    import importlib
    for mod in ["config", "aura.models", "aura.data_loader", "aura.split_manager",
                "aura.detector", "aura.local_training", "aura.root_gradient",
                "aura.fl_client", "aura.fl_server", "aura.merkle_tree"]:
        importlib.import_module(mod)


def synthetic_pipeline():
    import torch
    import config as cfg
    from aura.models import AURAModelBundle
    from aura.detector import AURAInferenceEngine
    from aura.local_training import run_two_pass_local_training
    from aura.fl_server import dc_fltrust_aggregate

    torch.manual_seed(0)
    bundle = AURAModelBundle()
    N, E, F = 20, 40, cfg.FEATURE_DIM

    engine = AURAInferenceEngine(bundle.autoencoder, bundle.stgnn)
    for i in range(cfg.EMA_WARMUP_BATCHES + 3):
        engine.process({
            "x": torch.rand(N, F), "edge_index": torch.randint(0, N, (2, E)),
            "edge_attr": torch.rand(E, F), "window_id": f"smoke:w{i}",
        })

    def deltas():
        ae, head = AURAModelBundle().autoencoder, AURAModelBundle().attack_head
        g_ae = {k: v.clone() for k, v in ae.state_dict().items()}
        g_head = {k: v.clone() for k, v in head.state_dict().items()}
        _, _, _, _, export = run_two_pass_local_training(
            ae, head, torch.rand(300, F),
            torch.optim.Adam(ae.parameters(), lr=1e-3),
            torch.optim.Adam(head.parameters(), lr=1e-3),
            mse_threshold=cfg.CH2_MSE_SPLIT_THRESHOLD,
        )
        ae_d = {k: export[k] - g_ae[k] for k in g_ae}
        head_d = {k: head.state_dict()[k] - g_head[k] for k in g_head}
        return ae_d, head_d

    clients = [deltas() for _ in range(3)]
    root_ae, root_head = deltas()
    dc_fltrust_aggregate(
        [c[0] for c in clients], [c[1] for c in clients], root_ae, root_head,
        client_round_counts=[1, 1, 1], ch2_warmup_rounds=cfg.CH2_WARMUP_ROUNDS,
    )


def dc_fltrust_safeguards():
    import torch
    from aura.fl_server import dc_fltrust_aggregate

    torch.manual_seed(0)
    root = {"w": torch.randn(200) * 0.01}
    honest = [{"w": root["w"] + torch.randn(200) * 0.01} for _ in range(4)]
    head = {"w": torch.zeros(4)}

    # A client scaling its update x100 cannot move the model further than the root does
    scaled = {"w": (0.3 * root["w"] + torch.randn(200) * 0.01) * 100}
    agg = dc_fltrust_aggregate(honest + [scaled], [head] * 5, root, head, [1] * 5, current_round=1)[0]
    assert agg["w"].norm() <= root["w"].norm() * 1.001, "scaled update was not bounded"

    # Channel 1 still excludes during channel 2 warmup
    reversed_client = {"w": -root["w"]}
    out = dc_fltrust_aggregate(honest + [reversed_client], [head] * 5, root, head, [1] * 5, current_round=1)
    assert out[5][4], "reversed client was not excluded during warmup"
    assert not any(out[5][:4]), "honest client was excluded during warmup"

    # Each channel gates only its own model
    root_head = {"w": torch.randn(50) * 0.01}
    ok_head = [{"w": root_head["w"] + torch.randn(50) * 0.002} for _ in range(5)]
    flipped_head = ok_head[:4] + [{"w": -ok_head[4]["w"]}]
    both = [{"w": root["w"] + torch.randn(200) * 0.01} for _ in range(5)]

    def run(ae_list, head_list, **kw):
        return dc_fltrust_aggregate(ae_list, head_list, root, root_head, [1] * 5, ch2_warmup_rounds=0, current_round=1, **kw)

    out = run(both, flipped_head)
    assert out[5][4], "flipped head was not flagged"
    assert torch.allclose(out[0]["w"], run(both, ok_head)[0]["w"]), "head score changed the AE aggregate"
    legacy = run(both, flipped_head, joint_exclusion=True)
    assert not torch.allclose(out[0]["w"], legacy[0]["w"]), "joint_exclusion did not restore the old rule"
    assert torch.allclose(out[1]["w"], legacy[1]["w"]), "flipped head reached the head aggregate"

    flipped_ae = both[:4] + [{"w": -root["w"]}]
    out = run(flipped_ae, ok_head)
    assert out[5][4], "flipped AE was not flagged"
    assert torch.allclose(out[1]["w"], run(both, ok_head)[1]["w"]), "AE score changed the head aggregate"


def client_and_root_share_a_recipe():
    import copy
    import torch
    import config as cfg
    from aura import dataset as ds, fl_round

    table, split = ds.load_flow_table(), ds.compute_split()
    flows = ds.flows_tensor(split.benign_flows("train", table)[:12000])
    model = fl_round.load_global_model()
    g_ae, g_head = fl_round.snapshot(model.autoencoder), fl_round.snapshot(model.attack_head)

    torch.manual_seed(cfg.SEED)
    update = fl_round.client_update(model.autoencoder, model.attack_head, g_ae, g_head, flows)
    # Building the root's model would consume random numbers and shift the batch
    # shuffle, so reuse an already built one for this comparison.
    build_ae, fl_round.FlowAutoencoder = fl_round.FlowAutoencoder, lambda: copy.deepcopy(model.autoencoder)
    try:
        torch.manual_seed(cfg.SEED)
        root_ae, _ = fl_round.root_reference(g_ae, g_head, flows, torch.empty(0))
    finally:
        fl_round.FlowAutoencoder = build_ae
    flat = lambda d: torch.cat([v.flatten() for v in d.values()])
    gap = float((flat(update.ae_delta) - flat(root_ae)).abs().max())
    assert gap < 1e-6, f"client and root AE updates differ by {gap} on identical data and seed"
    assert flat(root_ae).norm() > 0, "root update is empty"


def split_is_leakage_free():
    import numpy as np
    from aura import dataset as ds
    table = ds.load_flow_table()
    dev, full = ds.compute_split(0.3, table), ds.compute_split(1.0, table)   # both run verify_split()
    for name in ("test", "calib", "root"):
        assert np.array_equal(dev.windows[name], full.windows[name]), \
            f"{name} windows change with TRAIN_DATA_FRACTION"
    covered = np.concatenate([full.windows[k] for k in ("test", "calib", "root", "train_pool")])
    assert len(covered) == table.n_windows == len(np.unique(covered)), \
        "split sets do not cover every window exactly once"
    # A fresh split and the same split reloaded from disk must hash alike, or the manifest goes stale
    ds._SPLITS.clear()
    fresh = ds.compute_split(0.5, table).meta["hash"]
    ds._SPLITS.clear()
    assert ds.compute_split(0.5, table).meta["hash"] == fresh, "split hash changes after reloading"
    ds.compute_split(table=table)       # restore the configured split file


def manifest_matches():
    from aura import dataset as ds
    if not ds.MANIFEST_FILE.exists():
        raise FileNotFoundError("no manifest yet — run train.py")
    ds.check_manifest()


def one_federation_round():
    from aura.fl_server import run_federation_simulation
    results = run_federation_simulation(n_rounds=1)
    assert results and results[0].get("model_hash"), "no aggregated model returned"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--fl", action="store_true", help="also run one real federation round")
    args = ap.parse_args()

    print("AURA smoke test")
    check("all tracked .py files parse", parse_all_sources)
    check("config imports on a fresh clone", fresh_clone_config_import)
    check("core modules import", import_core_modules)
    check("synthetic models/detector/DC-FLTrust pipeline", synthetic_pipeline)
    check("DC-FLTrust bounds scaled updates, excludes during warmup, gates each model by its own channel", dc_fltrust_safeguards)
    check("canonical split is leakage-free and fraction-invariant", split_is_leakage_free)
    check("model artefacts match the manifest", manifest_matches)
    check("client and server root follow the same training recipe", client_and_root_share_a_recipe)
    if args.fl:
        check("one in-process federation round", one_federation_round)

    if _failures:
        print(f"\n{len(_failures)} check(s) failed.")
        sys.exit(1)
    print("\nAll checks passed.")


if __name__ == "__main__":
    main()
