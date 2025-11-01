#!/usr/bin/env python
"""
Utility script to evaluate TEAL-pruned models with lm-evaluation-harness tasks.

This wraps TEAL's sparse model loader and hands the resulting model object to
`lm_eval.models.huggingface.HFLM`, so we can reuse all existing HF-backed task
implementations without touching the harness core.

Usage:
export SPDP_HOME=/mnt/data/jhkim/spdp/spdp
python TEAL/run_lm_eval_teal.py \
    --model_name $SPDP_HOME/wanda/out/llama2_7b/unstructured/wanda0.45 \
    --teal_path $SPDP_HOME/TEAL/models/llama2_7b/unstructured/wanda0.45/histogram \
    --sparsity 0.55 \
    --tasks hellaswag,arc_easy,boolq \
    --batch_size 2 \
    --limit 100 \
    --output_json $SPDP_HOME/results/teal_lm_eval_55.json
"""

import argparse
import os
import sys
from typing import List, Optional

import json


def _resolve_spdp_home() -> str:
    spdp_home = os.environ.get("SPDP_HOME")
    if not spdp_home:
        raise EnvironmentError("SPDP_HOME environment variable is not set.")
    return spdp_home


def _import_dependencies(spdp_home: str) -> None:
    """Ensure TEAL + lm-eval modules are importable."""
    teal_root = os.path.join(spdp_home, "TEAL")
    lm_eval_root = os.path.join(spdp_home, "lm-evaluation-harness")

    for path in (teal_root, lm_eval_root):
        if path not in sys.path:
            sys.path.append(path)


def _parse_tasks(raw: str) -> List[str]:
    tasks = [task.strip() for task in raw.split(",") if task.strip()]
    if not tasks:
        raise ValueError("At least one task must be provided via --tasks.")
    return tasks


def _load_teal_model(args) -> tuple:
    """Instantiate tokenizer + sparse model configured with TEAL histograms."""
    import torch

    from utils.utils import get_tokenizer, get_sparse_model

    histogram_root = os.path.join(args.teal_path, "histograms")
    if not os.path.isdir(histogram_root):
        raise FileNotFoundError(
            f"Histogram directory not found: {histogram_root}. "
            "Run TEAL grab_acts.py beforehand."
        )

    tokenizer = get_tokenizer(args.model_name)

    model = get_sparse_model(
        args.model_name,
        device="auto" if args.device == "auto" else args.device,
        histogram_path=histogram_root,
        eval_sparsity=args.eval_sparsity,
    )

    if args.greedy_lookup:
        lookup_path = os.path.join(args.teal_path, "lookup")
        if not os.path.isdir(lookup_path):
            raise FileNotFoundError(
                f"Greedy lookup directory not found: {lookup_path}. "
                "Disable --greedy_lookup or generate greedy sparsities first."
            )
        model.load_greedy_sparsities(lookup_path, args.sparsity)
    else:
        model.set_uniform_sparsity(args.sparsity)

    model.eval()
    torch.set_grad_enabled(False)

    return tokenizer, model


def evaluate_with_lm_eval(args) -> dict:
    import torch

    from lm_eval import evaluator
    from lm_eval.models.huggingface import HFLM

    tasks = _parse_tasks(args.tasks)
    tokenizer, model = _load_teal_model(args)

    hf_lm = HFLM(
        pretrained=model,
        backend="causal",
        tokenizer=tokenizer,
        device=args.device if args.device != "auto" else ("cuda" if torch.cuda.is_available() else "cpu"),
        dtype=args.dtype,
        batch_size=args.batch_size,
        max_batch_size=args.max_batch_size,
        trust_remote_code=True,
        truncation=args.truncation,
        add_bos_token=args.add_bos_token,
    )

    results = evaluator.simple_evaluate(
        model=hf_lm,
        tasks=tasks,
        batch_size=args.batch_size,
        max_batch_size=args.max_batch_size,
        num_fewshot=args.num_fewshot,
        limit=args.limit,
        use_cache=args.use_cache,
        cache_requests=args.cache_requests,
        rewrite_requests_cache=args.rewrite_requests_cache,
        delete_requests_cache=args.delete_requests_cache,
        bootstrap_iters=args.bootstrap_iters,
        log_samples=args.log_samples,
        gen_kwargs=args.gen_kwargs,
        predict_only=args.predict_only,
        random_seed=args.random_seed,
        numpy_random_seed=args.numpy_random_seed,
        torch_random_seed=args.torch_random_seed,
        fewshot_random_seed=args.fewshot_random_seed,
    )

    return results


def _maybe_dump_results(results: dict, output_path: Optional[str]) -> None:
    if not output_path:
        return
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, "w") as fout:
        json.dump(results, fout, indent=2)


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Evaluate TEAL-pruned models with lm-evaluation-harness."
    )
    parser.add_argument(
        "--model_name",
        type=str,
        default="meta-llama/Llama-2-7b-hf",
        help="HF repo or local path to the (unstructured) base model weights.",
    )
    parser.add_argument(
        "--teal_path",
        type=str,
        required=True,
        help="Directory produced by TEAL grab_acts (contains histograms/).",
    )
    parser.add_argument(
        "--sparsity",
        type=float,
        default=0.5,
        help="TEAL dynamic sparsity to apply at evaluation time.",
    )
    parser.add_argument(
        "--tasks",
        type=str,
        required=True,
        help="Comma-separated list of lm-eval-harness task names.",
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=1,
        help="Per-device evaluation batch size.",
    )
    parser.add_argument(
        "--max_batch_size",
        type=int,
        default=None,
        help="Upper bound for automatic batch size search.",
    )
    parser.add_argument(
        "--num_fewshot",
        type=int,
        default=0,
        help="Number of few-shot examples per lm-eval task.",
    )
    parser.add_argument(
        "--limit",
        type=float,
        default=None,
        help="Optional limit on number (or fraction) of samples per task.",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cuda",
        help="Device string passed to HFLM (use 'auto' for device_map=auto).",
    )
    parser.add_argument(
        "--dtype",
        type=str,
        default="auto",
        help="Computation dtype hint for lm-eval harness ('auto', 'float16', ...).",
    )
    parser.add_argument(
        "--gen_kwargs",
        type=str,
        default=None,
        help="JSON or comma-separated generation args passed to lm-eval.",
    )
    parser.add_argument(
        "--output_json",
        type=str,
        default=None,
        help="Optional path to save raw lm-eval results as JSON.",
    )
    parser.add_argument(
        "--eval_sparsity",
        action="store_true",
        help="Enable TEAL overlapping sparsity logging (requires wandb auth).",
    )
    parser.add_argument(
        "--greedy_lookup",
        action="store_true",
        help="Load per-layer greedy sparsities instead of uniform masking.",
    )
    parser.add_argument(
        "--log_samples",
        action="store_true",
        help="Request per-sample outputs from lm-eval harness.",
    )
    parser.add_argument(
        "--cache_requests",
        action="store_true",
        help="Cache dataset requests to speed up repeated evaluations.",
    )
    parser.add_argument(
        "--rewrite_requests_cache",
        action="store_true",
        help="Force rewrite of existing cached dataset requests.",
    )
    parser.add_argument(
        "--delete_requests_cache",
        action="store_true",
        help="Delete cached dataset requests before evaluation starts.",
    )
    parser.add_argument(
        "--use_cache",
        type=str,
        default=None,
        help="Path to sqlite cache for model responses.",
    )
    parser.add_argument(
        "--bootstrap_iters",
        type=int,
        default=100000,
        help="Number of bootstrap iterations for stderr computation.",
    )
    parser.add_argument(
        "--predict_only",
        action="store_true",
        help="Skip metric computation and only dump model generations.",
    )
    parser.add_argument(
        "--random_seed",
        type=int,
        default=0,
        help="Seed for Python's random module.",
    )
    parser.add_argument(
        "--numpy_random_seed",
        type=int,
        default=1234,
        help="Seed for NumPy RNG.",
    )
    parser.add_argument(
        "--torch_random_seed",
        type=int,
        default=1234,
        help="Seed for torch RNG.",
    )
    parser.add_argument(
        "--fewshot_random_seed",
        type=int,
        default=1234,
        help="Seed for few-shot sampling RNG.",
    )
    parser.add_argument(
        "--truncation",
        action="store_true",
        help="Enable tokenizer truncation flag inside lm-eval HFLM wrapper.",
    )
    parser.add_argument(
        "--add_bos_token",
        action="store_true",
        help="Request BOS token insertion for causal decoding.",
    )
    return parser


def main():
    parser = build_arg_parser()

    if any(flag in sys.argv for flag in ("-h", "--help")):
        parser.parse_args()
        return

    spdp_home = _resolve_spdp_home()
    _import_dependencies(spdp_home)

    args = parser.parse_args()

    results = evaluate_with_lm_eval(args)
    _maybe_dump_results(results, args.output_json)

    print(json.dumps(results, indent=2))


if __name__ == "__main__":
    main()
