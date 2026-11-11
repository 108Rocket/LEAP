"""Path adapter for archived paper objectives and existing DTW trainers."""
import ast
import hashlib
import json
from pathlib import Path
import sys
import types

from .config import ROOT, checked_file, integer, resolve
from .logging import output_path

RECIPES = {
    "paper-probes": "Paper: 50 epochs + retain48 + 50 epochs + final placement",
    "dtw-probes": "Existing ordered-DTW-probe trainer: 5 epochs",
    "dtw-routing": "Existing DTW candidate-pool trainer: 100 epochs",
}


def plan(path, cfg, recipe, output=None):
    if recipe not in RECIPES:
        raise ValueError("Unknown training recipe")
    spec = cfg.get("training", {})
    if not spec.get("raw_inputs"):
        raise ValueError("Configure training.raw_inputs first; see 05_examples/reproduce_seed2000.json")
    if recipe == "paper-probes" and not spec.get("hausdorff_teacher"):
        raise ValueError("Configure training.hausdorff_teacher first")
    raw = resolve(path, spec["raw_inputs"])
    teacher = resolve(path, spec["hausdorff_teacher"]) if recipe == "paper-probes" else None
    dest = (Path(output).expanduser().resolve() if output else
            output_path(resolve(path, cfg.get("output_root", "repo:14_logs")), 'training', recipe))
    return dict(recipe=recipe, raw_inputs=str(raw), teacher=str(teacher) if teacher else None,
                trusted_teacher=spec.get("trusted_teacher") is True,
                seed=integer(spec.get("seed", 2000), "seed", 0, 2**31-1),
                device=cfg.get("device", "cuda:0"), output=str(dest),
                phases=RECIPES[recipe], status="adapter; no automatic training or test-based selection")


class SeedLiterals(ast.NodeTransformer):
    """Same seed substitution as the archived three-seed orchestrator."""
    def __init__(self, seed, expected=None):
        self.seed, self.expected = seed, expected

    def visit_Constant(self, node):
        if node.value == "ef91a5f89f4384fbe1b09273aef0c1dc4aed3106a0d2acc5406ee8da900935cf" and self.expected:
            return ast.copy_location(ast.Constant(self.expected), node)
        return node

    def visit_Call(self, node):
        self.generic_visit(node)
        if isinstance(node.func, ast.Attribute) and node.func.attr in ("manual_seed", "seed", "default_rng"):
            node.args = [ast.copy_location(ast.Constant(self.seed), a)
                         if isinstance(a, ast.Constant) and a.value == 2000 else a for a in node.args]
        return node

    def visit_Dict(self, node):
        self.generic_visit(node)
        for i, key in enumerate(node.keys):
            if isinstance(key, ast.Constant) and key.value == "seed":
                value = node.values[i]
                if isinstance(value, ast.Constant) and value.value == 2000:
                    node.values[i] = ast.copy_location(ast.Constant(self.seed), value)
        return node


def load_stage(name, path, seed, expected=None):
    source = path.read_text(encoding="utf-8")
    tree = ast.fix_missing_locations(SeedLiterals(seed, expected).visit(ast.parse(source)))
    module = types.ModuleType(name)
    module.__file__ = str(path)
    exec(compile(tree, str(path), "exec"), module.__dict__)
    return module, hashlib.sha256(source.encode()).hexdigest()


def paper(plan):
    import numpy as np
    import torch
    from .replay import save

    source = ROOT / "06_reproduction/training_sources"
    required = ("phase1_train.py", "phase2_train.py", "select_probes.py", "place.py",
                "learned_fields.py", "diagnostic_kernel.py")
    if not all((source / f).is_file() for f in required):
        raise ValueError("Paper training requires the source checkout (editable install)")
    dest = Path(plan["output"])
    dest.mkdir(parents=True, exist_ok=False)
    cfg = types.ModuleType("config")
    cfg.ROOT = dest / "phase1"
    cfg.RAW, cfg.TEACHER = Path(plan["raw_inputs"]), Path(plan["teacher"])
    cfg.NEGATIVE_RANKS = (1, 5, 9, 13, 17, 21, 25, 31)
    cfg.CONDITIONS = ("baseline", "protect48")
    cfg.WITNESS_WEIGHTS = {"baseline": 0.0, "protect48": 0.1}

    def setup():
        import numba
        torch.set_num_threads(4)
        numba.set_num_threads(min(4, numba.config.NUMBA_NUM_THREADS))
        torch.cuda.set_device(plan["device"])
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cudnn.allow_tf32 = False

    cfg.setup = setup
    old_config = sys.modules.get("config")
    old_path = sys.path[:]
    sys.path.insert(0, str(source))
    sys.modules["config"] = cfg
    save(dest / "entrance_protocol.json", plan)
    hashes = {}
    try:
        phase1, hashes["phase1"] = load_stage("entrance_phase1", source / "phase1_train.py", plan["seed"])
        phase1.main("baseline")
        cfg.START = cfg.ROOT / "training/baseline/latest.pt"
        warm = torch.load(cfg.START, weights_only=False, map_location="cpu")
        expected = hashlib.sha256(warm["state"]["anchors"].numpy().tobytes()).hexdigest()
        cfg.ROOT = dest / "phase2"
        cfg.ROOT.mkdir()
        selector, hashes["selection"] = load_stage("entrance_selector", source / "select_probes.py", plan["seed"])
        selector.main()
        phase2, hashes["phase2"] = load_stage("entrance_phase2", source / "phase2_train.py", plan["seed"], expected)
        phase2.main("protect48")
        payload = json.loads((cfg.ROOT / "training/protect48/epoch_050.json").read_text())
        anchors = np.asarray(payload["state"]["anchors"], np.float32)
        placement = dest / "placement"
        placement.mkdir()
        asset = placement / "protected_asset.json"
        save(asset, dict(anchors=anchors.tolist(), anchors_sha256=hashlib.sha256(anchors.tobytes()).hexdigest(),
                         geometry_mean=payload["mean"], geometry_std=payload["std"]))
        place, hashes["placement"] = load_stage("entrance_placement", source / "place.py", plan["seed"])
        place.ROOT, place.RAW, place.TEACHER = placement, cfg.RAW, cfg.TEACHER
        original = place.DualFieldRetriever

        class BoundRetriever:
            @staticmethod
            def from_asset():
                return original.from_asset(asset, device=plan["device"])

        place.DualFieldRetriever = BoundRetriever
        place.main()
        save(dest / "ALL_DONE.json", dict(final_asset=str(placement / "assets/exact_region.json"),
                                         seed=plan["seed"], source_sha256=hashes, test_selection_enabled=False))
    except Exception as exc:
        save(dest / "FAILED.json", {"error": str(exc)})
        raise
    finally:
        sys.path[:] = old_path
        if old_config is None:
            sys.modules.pop("config", None)
        else:
            sys.modules["config"] = old_config


def run(plan):
    import platform
    import torch
    if platform.system() != "Linux" or not torch.cuda.is_available():
        raise RuntimeError("Training requires the Linux CUDA data host")
    torch.cuda.set_device(plan["device"])
    plan["raw_sha256"] = checked_file({"path": plan["raw_inputs"]})
    if plan["teacher"]:
        if not plan["trusted_teacher"]:
            raise ValueError("The historical teacher is a pickle; set trusted_teacher=true only for trusted data")
        plan["teacher_sha256"] = checked_file({"path": plan["teacher"]})
    if Path(plan["output"]).exists():
        raise FileExistsError("Use a new output directory; training is never overwritten")
    if plan["recipe"] == "paper-probes":
        paper(plan)
        return
    module = "leap.training.train_dtw_probes" if plan["recipe"] == "dtw-probes" else "leap.training.train_dtw_routing"
    import importlib
    importlib.import_module(module).main(["--raw-inputs", plan["raw_inputs"], "--output", plan["output"],
                                        "--device", plan["device"], "--seed", str(plan["seed"]),
                                        "--epochs", "5" if plan["recipe"] == "dtw-probes" else "100"])
