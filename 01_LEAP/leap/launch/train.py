"""Isolated SDD near/far adapter for the archived two-phase training recipe.

Run only in a fresh subprocess: the archived stages import their dependencies
at module scope, so this adapter binds their point-set operations before loading.
"""
import argparse
import ast
import hashlib
import json
from pathlib import Path
import sys
import types

from .files import save, sha256
from .data import prepare_training, pointset_hd
from .wizard import ROOT


class Sizes(ast.NodeTransformer):
    def __init__(self, n, nv):
        self.n, self.nv = n, nv

    def visit_Assert(self, node):
        if ast.unparse(node.test) == "len(pool) == 1024":
            node.test = ast.parse("len(pool) >= 16", mode="eval").body
            return node
        return self.generic_visit(node)

    def visit_Call(self, node):
        if (isinstance(node.func, ast.Attribute) and node.func.attr == "choice" and node.args
                and isinstance(node.args[0], ast.Constant) and node.args[0].value == 1024):
            node.args[0] = ast.parse("len(pool)", mode="eval").body
        return self.generic_visit(node)

    def visit_List(self, node):
        if all(isinstance(v, ast.Constant) for v in node.elts) and [v.value for v in node.elts] == [7000, 1000, 2000]:
            return ast.copy_location(ast.List(elts=[ast.Constant(v) for v in (self.n, self.nv, 0)], ctx=ast.Load()), node)
        return self.generic_visit(node)

    def visit_Constant(self, node):
        replacements = {7000: self.n, 1024: min(1024, self.n), 5500: 50 * ((self.n + 63) // 64)}
        if type(node.value) is int and node.value in replacements:
            return ast.copy_location(ast.Constant(replacements[node.value]), node)
        return node


def nearfar_source(name, text):
    """Match the accepted same-field selection/placement adaptation exactly."""
    replacements = {
        "select_probes": [("torch.maximum(qb[:,None]-da,db-qa[:,None]).clamp_min(0)",
                           "torch.maximum((qa[:,None]-da).abs(),(qb[:,None]-db).abs())")],
        "place": [("tl.maximum(tl.maximum(qb-da,db-qa),0.)", "tl.maximum(tl.abs(qa-da),tl.abs(qb-db))"),
                  ("torch.maximum(qb-da,dbb-qa).diagonal().clamp_min(0)",
                   "torch.maximum((qa-da).abs(),(qb-dbb).abs()).diagonal()")],
    }
    for old, new in replacements.get(name, []):
        if text.count(old) != 1:
            raise ValueError("Archived training source changed; review near/far adaptation: " + name)
        text = text.replace(old, new)
    return text


def run(source, output, device="cuda:0", preflight_only=False):
    import numpy as np
    import numba
    import torch
    from leap.common import model, data as geometry, dual_field
    from leap.training import data as training_data
    from leap.hausdorff.pointset_index import fields
    from leap.common.data import checked_pack
    from leap.common.raw import normalize_points
    from leap.common.model import TrajectoryIndex
    from leap.reproduction.training import SeedLiterals

    output = Path(output)
    output.mkdir(parents=True, exist_ok=False)
    if not torch.cuda.is_available():
        raise RuntimeError("Training requires a Linux CUDA host")
    torch.cuda.set_device(device)
    torch.set_num_threads(4)
    numba.set_num_threads(min(4, numba.config.NUMBA_NUM_THREADS))
    torch.manual_seed(2000)
    np.random.seed(2000)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    prepared = prepare_training(source, output / "inputs")
    with np.load(prepared / "raw_inputs.npz", allow_pickle=False) as z:
        paths = []
        for prefix in ("train", "validation"):
            points, offsets = z[prefix + "_points"], z[prefix + "_offsets"]
            paths.append([points[a:b] for a, b in zip(offsets[:-1], offsets[1:])])
        paths.append([])
        mean, std = z["geometry_mean"], z["geometry_std"]
    tm = np.load(prepared / "haus_train.npy", allow_pickle=False) / std[0]
    vm = np.load(prepared / "haus_validation.npy", allow_pickle=False) / std[0]
    labels = (torch.tensor(tm, dtype=torch.float32), torch.tensor(vm, dtype=torch.float32), torch.empty((0, 0)), 1.)
    n, nv = len(paths[0]), len(paths[1])
    ids = [int(tm.mean(1).argmin())]
    nearest = tm[ids[0]].copy()
    nearest[ids] = -np.inf
    while len(ids) < 64:
        pick = int(nearest.argmax())
        ids.append(pick)
        nearest = np.minimum(nearest, tm[pick])
        nearest[ids] = -np.inf
    probes = []
    for i in ids:
        p = ((paths[0][i] - mean) / std).astype(np.float32)
        edge = p[-1] - p[0]
        t = np.clip(((p - p[0]) * edge).sum(1) / max(float((edge * edge).sum()), 1e-30), 0, 1)
        residual = ((p - p[0] - t[:, None] * edge)**2).sum(1)
        mid = int(residual[1:-1].argmax()) + 1 if len(p) > 2 else 0
        probes.append(p[[0, mid, len(p) - 1]])
    probes = np.asarray(probes, np.float32)
    initial = dict(initial_probes=probes.tolist(), geometry_mean=mean.tolist(), geometry_std=std.tolist())
    model.load_probe_asset = lambda path=None: initial
    training_data.load_raw_splits = lambda path, asset: paths
    geometry.load_distances = lambda path: labels
    archive = ROOT / "06_reproduction/training_sources"
    sys.path[:0] = [str(archive), str(archive / "sdd")]
    from sdd_point_fields import install, pair_stats
    install()

    def differentiable(points, lengths, anchors):
        rows = []
        for lo in range(0, len(points), 16):
            p, length = points[lo:lo + 16], lengths[lo:lo + 16]
            width = int(length.max())
            d = (p[:, :width, None] - anchors[None, None]).square().sum(-1)
            mask = torch.arange(width, device=p.device)[None] < length[:, None]
            near = d.masked_fill(~mask[:, :, None], torch.inf).amin(1).clamp_min(1e-24).sqrt()
            far = d.masked_fill(~mask[:, :, None], -torch.inf).amax(1).clamp_min(1e-24).sqrt()
            rows.append(torch.cat((near, far), -1))
        return torch.cat(rows)

    def scan(q, d, same=False):
        from leap.hausdorff.pointset_index import scan as point_scan
        result = point_scan(q, d, nearfar=True)
        if same:
            result.fill_diagonal_(torch.inf)
        return result

    @torch.no_grad()
    def encode(self, trajectories):
        points, offsets = checked_pack(trajectories)
        p = torch.from_numpy(normalize_points(points, self.mean, self.std)).to(device)
        o = torch.from_numpy(offsets).to(device)
        return TrajectoryIndex(fields(p, o, self.anchors, True), p, o, self.asset_id + ":pointset-nearfar")

    dual_field.fields_packed = lambda p, o, a: fields(p, o, a, True)
    dual_field.fields_torch = differentiable
    dual_field.score_torch = lambda q, d: (q - d).abs().amax(-1)
    dual_field.scan = scan
    dual_field.DualFieldRetriever.encode = encode
    import diagnostic_kernel
    diagnostic_kernel.pair_stats = pair_stats
    from learned_fields import LearnedFields
    net = LearnedFields(probes[:, 1], mean, std, "radial")
    index = net.encode(paths[0])
    q = torch.arange(16, device=device)
    d = (q * 17 + 31) % n
    raw, _ = pair_stats(index, None, q, d)
    torch.testing.assert_close(raw[:, 0], labels[0].to(device)[q, d], rtol=2e-5, atol=2e-5)
    lengths = index.offsets[1:] - index.offsets[:-1]
    padded = torch.zeros((16, int(lengths[:16].max()), 2), device=device)
    for i in range(16):
        p = index.points[index.offsets[i]:index.offsets[i + 1]]
        padded[i, :len(p)] = p
    value = net(padded, lengths[:16])
    torch.testing.assert_close(value, index.coordinates[:16], rtol=2e-5, atol=2e-6)
    value.sum().backward()
    if not torch.isfinite(net.anchors.grad).all() or net.anchors.grad.norm() == 0:
        raise ValueError("Training preflight gradient failed")
    save(output / "PREFLIGHT.json", {"train": n, "validation": nv, "seed": 2000,
        "initial_source_ids": ids, "teacher_parity": True, "packed_forward_parity": True,
        "finite_nonzero_gradient": True, "training_completed": False})
    cfg = types.ModuleType("config")
    cfg.ROOT, cfg.RAW, cfg.TEACHER = output / "phase1", prepared / "raw_inputs.npz", prepared / "haus_train.npy"
    cfg.START = cfg.ROOT / "training/baseline/latest.pt"
    cfg.NEGATIVE_RANKS = (1, 5, 9, 13, 17, 21, 25, 31)
    cfg.CONDITIONS = ("baseline", "protect48")
    cfg.WITNESS_WEIGHTS = {"baseline": 0., "protect48": .1}
    cfg.setup = lambda: None
    sys.modules["config"] = cfg
    hashes = {}

    def load(name, expected=None):
        path = archive / (name + ".py")
        text = nearfar_source(name, path.read_text(encoding="utf-8"))
        tree = Sizes(n, nv).visit(ast.parse(text))
        tree = ast.fix_missing_locations(SeedLiterals(2000, expected).visit(tree))
        # Triton inspects source files, so transformed kernels need real files.
        runtime = output / "runtime"
        runtime.mkdir(exist_ok=True)
        adapted = runtime / (name + ".py")
        adapted.write_text(ast.unparse(tree) + "\n", encoding="utf-8")
        hashes[name] = {"source_sha256": sha256(path), "adapted_source_sha256": sha256(adapted)}
        save(output / "SOURCE_ADAPTERS.json", hashes)
        module = types.ModuleType("porto_" + name)
        module.__file__ = str(adapted)
        sys.modules[module.__name__] = module
        exec(compile(adapted.read_text(encoding="utf-8"), str(adapted), "exec"), module.__dict__)
        return module

    if preflight_only:
        for name in ("phase1_train", "phase2_train", "select_probes"):
            load(name)
        placement = load("place")
        with torch.no_grad():
            measured, witness = pair_stats(index, None, q, d)
            h = measured[:, 0].contiguous()
            target = (h * .8).contiguous()
            exact, _ = placement.coverage_matrix(index.coordinates, net.anchors, q, d,
                index.points[witness.long()].contiguous(), h, target)
            qa, qb = index.coordinates[q].chunk(2, -1)
            da, db = index.coordinates[d].chunk(2, -1)
            expected = torch.maximum((qa - da).abs(), (qb - db).abs()) > (target + 2e-5 * (1 + target.abs()))[:, None]
            torch.testing.assert_close(exact, expected, rtol=0, atol=0)
        save(output / "STAGE_PREFLIGHT.json", {"phase_sources_loaded": True,
            "nearfar_placement_kernel_matches_torch": True, "epochs_completed": 0})
        return
    del net, index, raw, padded, value
    torch.cuda.empty_cache()
    print("TRAINING phase 1: 50 epochs", flush=True)
    load("phase1_train").main("baseline")
    cfg.START = cfg.ROOT / "training/baseline/latest.pt"
    warm = torch.load(cfg.START, map_location="cpu", weights_only=False)
    if warm["epoch"] != 50:
        raise ValueError("Phase 1 did not finish all 50 epochs")
    expected = hashlib.sha256(warm["state"]["anchors"].numpy().tobytes()).hexdigest()
    cfg.ROOT = output / "phase2"
    cfg.ROOT.mkdir()
    load("select_probes").main()
    print("TRAINING phase 2: 50 epochs", flush=True)
    load("phase2_train", expected).main("protect48")
    payload = json.loads((cfg.ROOT / "training/protect48/epoch_050.json").read_text())
    anchors = np.asarray(payload["state"]["anchors"], np.float32)
    asset = output / "protected_asset.json"
    save(asset, {"anchors": anchors.tolist(), "anchors_sha256": hashlib.sha256(anchors.tobytes()).hexdigest(),
                 "geometry_mean": mean.tolist(), "geometry_std": std.tolist()})
    placement = load("place")
    placement.ROOT, placement.RAW, placement.TEACHER = output / "placement", cfg.RAW, cfg.TEACHER
    retriever = placement.DualFieldRetriever
    placement.DualFieldRetriever = types.SimpleNamespace(from_asset=lambda: retriever.from_asset(asset, device=device))
    placement.cpu_h = pointset_hd
    placement.main()
    final = json.loads((placement.ROOT / "assets/exact_region.json").read_text())
    final.update(city="Porto", supervision_distance="SDD point-set Hausdorff", seed=2000,
                 representation="pointset-nearfar-v1", epochs=100,
                 split_protocol_sha256=sha256(prepared / "protocol.json"), test_selection_enabled=False)
    save(output / "final_asset.json", final)
    save(output / "ALL_DONE.json", {"status": "trained", "epochs": 100,
                                   "asset_sha256": sha256(output / "final_asset.json")})


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--source-dir", required=True, type=Path)
    ap.add_argument("--output", required=True, type=Path)
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--preflight-only", action="store_true")
    args = ap.parse_args()
    try:
        run(args.source_dir, args.output, args.device, args.preflight_only)
    except BaseException as exc:
        if args.output.is_dir():
            save(args.output / "FAILED.json", {"type": type(exc).__name__, "message": str(exc)})
        raise


if __name__ == "__main__":
    main()
