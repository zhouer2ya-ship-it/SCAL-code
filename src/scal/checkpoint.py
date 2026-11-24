"""Validate resumable runs and restore the fitted HCBP warm-start state."""

from __future__ import annotations

import ast
from dataclasses import asdict, is_dataclass
import hashlib
import inspect
import json
from pathlib import Path
import platform
import textwrap

import numpy as np
import scipy
import sklearn

from .io import digest, json_value, read_json, write_json, write_npz
from .solving.HCBP_solver import HCBPSolver
from .solving.get_basis import basis_definition


def signature(value):
    return hashlib.sha256(
        json.dumps(json_value(value), sort_keys=True, allow_nan=False).encode("utf-8")
    ).hexdigest()


class _ExecutableSource(ast.NodeTransformer):
    """Keep executable syntax when fingerprinting a Python implementation."""

    def visit(self, node):
        node = super().visit(node)
        if (
            isinstance(node, (ast.Module, ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef))
            and node.body
            and isinstance(node.body[0], ast.Expr)
            and isinstance(node.body[0].value, ast.Constant)
            and isinstance(node.body[0].value.value, str)
        ):
            node.body = node.body[1:]
        return node


def source_signature(source):
    # Comments, formatting and docstrings can change while a run is in progress.
    # Executable syntax identifies the calculation that produced the saved model.
    tree = _ExecutableSource().visit(ast.parse(textwrap.dedent(source)))
    return signature(ast.dump(tree, include_attributes=False))


def callable_identity(function):
    """Identify source code; callers must identify model data or mutable globals."""
    target = function.__func__ if inspect.ismethod(function) else function
    if not inspect.isroutine(target):
        target = type(function)
    try:
        source = inspect.getsource(target)
    except (OSError, TypeError):
        source = None
    try:
        source_hash = source_signature(source) if source is not None else None
    except SyntaxError:
        # inspect can return a partial surrounding expression for inline lambdas.
        source_hash = signature(source)
    identity = {
        "module": target.__module__,
        "name": getattr(
            function, "__qualname__", getattr(function, "__name__", target.__qualname__)
        ),
        "source_sha256": source_hash,
    }
    # Bound methods also depend on their model parameters, such as beam width.
    owner = getattr(function, "__self__", function)
    if is_dataclass(owner) and not isinstance(owner, type):
        identity["parameters"] = json_value(asdict(owner))
    return identity


def runtime_identity():
    root = Path(__file__).parent
    paths = sorted(
        [*root.glob("solving/*.py"), *root.glob("sampling/*.py")]
        + [root / "coordinates.py", root / "acquisition.py", root / "functions.py"]
    )
    return {
        "python": platform.python_version(),
        "numpy": np.__version__,
        "scipy": scipy.__version__,
        "sklearn": sklearn.__version__,
        "numerical_code": signature(
            {
                p.relative_to(root).as_posix(): source_signature(p.read_text(encoding="utf-8-sig"))
                for p in paths
            }
        ),
    }


class FunctionCheckpoint:
    def __init__(self, store, settings, resume):
        self.store = store
        self.root = store.directory
        settings = json_value(settings)
        if resume:
            saved = read_json(self.root / "settings.json")
            if saved.get("checkpoint_schema") != 1:
                raise ValueError("Compatible checkpoint metadata is missing from settings.json")
            # Waiting limits may change; the physical response identity may not.
            saved.pop("response_options", None)
            comparable = {
                key: value for key, value in settings.items() if key != "response_options"
            }
            if saved != comparable:
                raise ValueError(
                    "Resume settings differ: keep the original model, distributions, seed, "
                    "budget, sampler and numerical environment"
                )
        else:
            store.create(settings)
        budgets = sorted(int(path.stem[1:]) for path in (self.root / "fits").glob("N*.json"))
        if budgets and budgets != list(range(settings["run"]["initial"], budgets[-1] + 1)):
            raise ValueError("Missing committed fit in the saved run")

    def inputs(self, space, config):
        path = self.root / "generated_inputs.npz"
        commit = self.root / "generated_inputs.json"
        if commit.exists():
            if digest(path) != read_json(commit)["sha256"]:
                raise ValueError("Generated inputs checksum mismatch")
        else:
            if (self.root / "initial_responses.npz").exists() or any(
                (self.root / "fits").glob("*.json")
            ):
                raise ValueError("Missing generated-input commit in an existing run")
            # No response is requested before this deterministic input commit.
            first, pool = np.random.SeedSequence(config.seed).spawn(2)
            write_npz(
                path,
                initial_X=space.sample(config.initial, np.random.default_rng(first)),
                candidate_pool=space.sample(config.pool_size, np.random.default_rng(pool)),
                candidate_ids=np.arange(config.pool_size, dtype=np.int64),
            )
            write_json(commit, {"sha256": digest(path)})
        with np.load(path, allow_pickle=False) as values:
            return tuple(
                values[key].copy() for key in ("initial_X", "candidate_pool", "candidate_ids")
            )

    def initial_responses(self, inputs, evaluate):
        # The external adapter commits each job during this batch. A partial
        # batch therefore recovers its completed rows from the per-point cache.
        path = self.root / "initial_responses.npz"
        commit = self.root / "initial_responses.json"
        if commit.exists():
            if digest(path) != read_json(commit)["sha256"]:
                raise ValueError("Initial responses checksum mismatch")
            with np.load(path, allow_pickle=False) as values:
                responses = values["y"].copy()
        elif path.exists():
            # Atomic NPZ survived an interruption immediately before its commit.
            with np.load(path, allow_pickle=False) as values:
                responses = values["y"].copy()
            write_json(commit, {"sha256": digest(path)})
        else:
            if any((self.root / "fits").glob("*.json")):
                raise ValueError("Missing initial responses in an existing run")
            responses = evaluate(inputs)
            write_npz(path, y=responses)
            write_json(commit, {"sha256": digest(path)})
        if responses.shape != (len(inputs),) or not np.isfinite(responses).all():
            raise ValueError("Invalid saved initial responses")
        return responses

    def load_fit(self, budget, config, inputs, responses):
        path = self.root / "fits" / f"N{budget:04d}.json"
        if not path.exists():
            return None, None
        record = read_json(path)
        if record.get("training_sha256") != signature({"X": inputs, "y": responses}):
            raise ValueError(f"Training observations changed at N={budget}")
        if digest(path.with_suffix(".npz")) != record.get("fit_sha256"):
            raise ValueError(f"Fit checksum mismatch at N={budget}")
        if digest(self.root / "basis.json") != record.get("basis_sha256"):
            raise ValueError("Basis checksum mismatch")
        basis = read_json(self.root / "basis.json")
        # The saved support and corrected error preserve HCBP's next-round
        # decision; a fresh fit here could change the subsequent selected point.
        model = HCBPSolver(np.asarray(basis["poly_matrix"]), max_degree=config.degree, q=config.q)
        model.mapping_, model.gamma_ = basis["mapping"], np.asarray(basis["gamma"])
        model.basis_definition_ = basis_definition(model.poly_matrix, config.degree, config.q)
        with np.load(path.with_suffix(".npz"), allow_pickle=False) as values:
            model.coef_, model.support_indices_ = values["coef"].copy(), values["support"].copy()
            for name, key in (
                ("mean_", "mean"),
                ("std_", "std"),
                ("error_", "error"),
                ("press_error_", "raw_PRESS"),
                ("ic_", "IC"),
            ):
                setattr(model, name, float(values[key]))
        model.corrected_error_ = model.error_
        model.raw_error_ = model.press_error_
        model.X, model.Y = inputs.copy(), responses.reshape(-1, 1).copy()
        model.num_basis = len(model.support_indices_)
        model.n_features_in_, model.is_fitted_ = inputs.shape[1], True
        model.warm_start_used_ = record["previous_selected"]
        return model, record

    def save_fit(self, model, record):
        basis = self.root / "basis.json"
        if not basis.exists():
            write_json(
                basis,
                {
                    "mapping": model.mapping_,
                    "gamma": model.gamma_,
                    "poly_matrix": model.poly_matrix,
                },
            )
        record["basis_sha256"] = digest(basis)
        record["training_sha256"] = signature({"X": model.X, "y": model.Y.ravel()})
        self.store.save_fit(model, record)

    def save_record(self, record):
        write_json(self.root / "fits" / f"N{record['budget']:04d}.json", record)
