import os
import json
import random
import hashlib
import re 
import subprocess
import sys
import tempfile
import ast
import time
import statistics
import shutil

from tqdm import tqdm
from openai import OpenAI
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

from lcb_runner.runner.parser import get_args
from lcb_runner.utils.scenarios import Scenario
from lcb_runner.lm_styles import LanguageModelStore
from lcb_runner.runner.runner_utils import build_runner
from lcb_runner.utils.path_utils import get_output_path
from lcb_runner.evaluation import extract_instance_results
from lcb_runner.runner.scenario_router import (
    build_prompt_benchmark,
    combine_results,
    sort_and_extract_save_results,
    get_metrics,
)


@dataclass
class EvalResult:
    ok: bool                 
    passed_all: bool
    passed_mask: List[bool]
    failed_test_ids: List[str]
    pass_fraction: float
    error_kind: Optional[str] = None  # "RE" / "TLE" / None
    error_detail: Optional[str] = None


def extract_property_test_benchmarks():
    args = get_args()

    model = LanguageModelStore[args.model]
    benchmarks, format_prompt = build_prompt_benchmark(args)

    benchmarks = benchmarks[:20]
    # NUM_BENCHMARKS = 10
    # random.seed(123)
    # benchmarks = random.sample(benchmarks, NUM_BENCHMARKS)

    output_path = f"output/{model.model_repr}_property_test_benchmarks.json"
    if os.path.exists(output_path):
        os.remove(output_path)

    runner = build_runner(args, model)
    results: list[list[str]] = runner.run_main(benchmarks, format_prompt)

    combined_results = combine_results(
        args.scenario, results, model, args.cot_code_execution
    )

    save_results = [
        instance.insert_output(outputs_list, extracted_list)
        for instance, (outputs_list, extracted_list) in zip(
            benchmarks, combined_results
        )
    ]

    save_results, combined_results = sort_and_extract_save_results(
        args.scenario, save_results
    )

    metrics = get_metrics(args.scenario, args, benchmarks, combined_results)
    graded = extract_instance_results(metrics[1])

    metadatas = metrics[2]

    # for instance, (_, extracted_list), graded_list, meta in zip(benchmarks, combined_results, graded, metadatas):
        # for i, (extracted_code, code_grade, code_meta) in enumerate(zip(extracted_list, graded_list, meta)):
            # path = f"output/generated_code/{instance.question_id}_{i}_code.py"
            # with open(path, "w") as f:
            #     f.write(f"'''\n{instance.question_content}\n\npassed: {code_grade}\n\nmetadata: {code_meta}\n'''\n\n")
            #     f.write(extracted_code)

    save_eval_results = [
        instance.insert_output_evaluation(
            outputs_list, extracted_list, graded_list, metadata=meta
        )
        for instance, (outputs_list, extracted_list), graded_list, meta in zip(
            benchmarks, combined_results, graded, metadatas
        )
    ]

    with open(output_path, "w") as f:
        json.dump(save_eval_results, f, indent=4)

def main():
    args = get_args()

    model = LanguageModelStore[args.model]
    benchmark, format_prompt = build_prompt_benchmark(args)
    if args.debug:
        print(f"Running with {len(benchmark)} instances in debug mode")
        benchmark = benchmark[:15]

    benchmark = benchmark[:15]
    # NUM_BENCHMARKS = 10
    # random.seed(123)
    # benchmark = random.sample(benchmark, NUM_BENCHMARKS)
    # benchmark = benchmark[:NUM_BENCHMARKS]

    output_path = get_output_path(model.model_repr, args)
    eval_file = output_path.replace(".json", "_eval.json")
    eval_all_file = output_path.replace(".json", "_eval_all.json")

    # if args.continue_existing or args.continue_existing_with_eval:
    #     if os.path.exists(output_path):
    #         with open(output_path, "r") as f:
    #             old_save_results = json.load(f)
    #     elif os.path.exists(eval_all_file):
    #         with open(eval_all_file, "r") as f:
    #             old_save_results = json.load(f)
    #     else:
    #         print(
    #             f"File {output_path} does not exist in --continue_existing, starting from scratch"
    #         )
    #         old_save_results = []

    #     old_save_results = [
    #         instance
    #         for instance in old_save_results
    #         if instance["output_list"] and [x for x in instance["output_list"] if x]
    #     ]
    #     old_save_results_question_ids = [
    #         instance["question_id"] for instance in old_save_results
    #     ]
    #     remaining_benchmark = [
    #         instance
    #         for instance in benchmark
    #         if instance.question_id not in old_save_results_question_ids
    #     ]
    #     print(
    #         f"Found {len(old_save_results)} existing generations, continuing with {len(remaining_benchmark)} remaining"
    #     )
    # else:
    if True:
        old_save_results = []
        remaining_benchmark = benchmark

    if len(remaining_benchmark) > 0:
        runner = build_runner(args, model)
        results: list[list[str]] = runner.run_main(remaining_benchmark, format_prompt)
    else:
        results = []

    combined_results = combine_results(
        args.scenario, results, model, args.cot_code_execution
    )

    save_results = [
        instance.insert_output(outputs_list, extracted_list)
        for instance, (outputs_list, extracted_list) in zip(
            remaining_benchmark, combined_results
        )
    ]

    # TODO: where do we run/test the generated code?

    if args.continue_existing or args.continue_existing_with_eval:
        save_results += old_save_results

    save_results, combined_results = sort_and_extract_save_results(
        args.scenario, save_results
    )

    with open(output_path, "w") as f:
        json.dump(save_results, f, indent=4)


    if args.evaluate:
        if args.continue_existing_with_eval and os.path.exists(eval_all_file):
            with open(eval_all_file) as fp:
                old_eval_all_results = json.load(fp)

            if os.path.exists(eval_file):
                with open(eval_file) as fp:
                    old_eval_results = json.load(fp)
            else:
                old_eval_results = None

            old_eval_results_question_ids = [
                instance["question_id"] for instance in old_eval_all_results
            ]
            remaining_indices = [
                idx
                for idx in range(len(benchmark))
                if benchmark[idx].question_id not in old_eval_results_question_ids
            ]
            benchmark = [benchmark[idx] for idx in remaining_indices]
            combined_results = [combined_results[idx] for idx in remaining_indices]

            old_eval_size = len(old_eval_results_question_ids)
            new_eval_size = len(benchmark)

            if new_eval_size == 0:
                return

            print(f"Found {old_eval_size}, running evals for {new_eval_size} problems")

            metrics = get_metrics(args.scenario, args, benchmark, combined_results)
            graded = extract_instance_results(metrics[1])

            if old_eval_results:
                for key in metrics[0]:
                    if key in old_eval_results[0]:
                        if key != "detail":
                            metrics[0][key] = (
                                old_eval_size * old_eval_results[0][key]
                                + new_eval_size * metrics[0][key]
                            )
                            metrics[0][key] /= old_eval_size + new_eval_size

                for key in metrics[0]["detail"]:
                    if key in old_eval_results[0]["detail"]:
                        metrics[0]["detail"][key] = {
                            **metrics[0]["detail"][key],
                            **old_eval_results[0]["detail"][key],
                        }
                metrics[1] = {**metrics[1], **old_eval_results[1]}
            else:
                print("Old eval file not present, cannot update eval file")
                metrics = {}

        else:
            metrics = get_metrics(args.scenario, args, benchmark, combined_results)
            graded = extract_instance_results(metrics[1])
            old_eval_all_results = []
            old_eval_results = []

        if args.scenario == Scenario.codegeneration:
            if metrics:
                metadatas = metrics[2]
            else:
                metadatas = [[] for _ in benchmark]
            save_eval_results = [
                instance.insert_output_evaluation(
                    outputs_list, extracted_list, graded_list, metadata=meta
                )
                for instance, (outputs_list, extracted_list), graded_list, meta in zip(
                    benchmark, combined_results, graded, metadatas
                )
            ]
            if metrics and old_eval_results:
                old_eval_results
                metrics[2] = old_eval_results[2] + metrics[2]
        elif args.scenario == Scenario.selfrepair:
            metadatas = metrics[2]
            with open(
                f"output/{model.model_repr}/{Scenario.codegeneration}_{args.codegen_n}_{args.temperature}_eval_all.json"
            ) as f:
                code_gen_evals = json.load(f)
            original_code_lists = [
                code_gen_eval["code_list"] for code_gen_eval in code_gen_evals
            ]

            save_eval_results = [
                instance.insert_output_evaluation(
                    outputs_list,
                    extracted_list,
                    graded_list,
                    metadata=meta,
                    original_code_list=original_code_list,
                )
                for instance, (
                    outputs_list,
                    extracted_list,
                ), graded_list, meta, original_code_list in zip(
                    benchmark, combined_results, graded, metadatas, original_code_lists
                )
            ]

        else:
            save_eval_results = [
                instance.insert_output_evaluation(
                    outputs_list, extracted_list, graded_list
                )
                for instance, (outputs_list, extracted_list), graded_list in zip(
                    benchmark, combined_results, graded
                )
            ]

        save_eval_results = old_eval_all_results + save_eval_results

        with open(eval_file, "w") as f:
            json.dump(metrics, f, indent=4)

        with open(eval_all_file, "w") as f:
            json.dump(save_eval_results, f, indent=4)



# def test() -> None:
#     args = get_args

#     # Import here to avoid hard dependency errors if user just wants to inspect code.
#     from datasets import load_dataset

#     ds = load_dataset(args.dataset, version_tag=args.version_tag)
#     # pick a split
#     if args.split:
#         data = ds[args.split]
#     else:
#         # common: only one split
#         split0 = list(ds.keys())[0]
#         data = ds[split0]

#     llm_map = load_llm_solutions_jsonl(args.llm_solutions_jsonl)

#     out_records: List[Dict[str, Any]] = []
#     n = len(data) if args.max_tasks is None else min(len(data), args.max_tasks)

#     for idx in range(n):
#         item = dict(data[idx])

#         qid_key, qid = first_present(item, ["question_id", "id", "task_id"])
#         if qid is None:
#             # fallback: stable-ish synthetic id
#             qid = f"idx_{idx}"

#         _, desc = first_present(item, ["question_content", "prompt", "description", "problem_statement"])
#         desc = desc if desc is not None else ""

#         # test cases
#         _, public_tc_raw = first_present(item, ["public_test_cases", "public_tests", "public_testcases", "public"])
#         _, private_tc_raw = first_present(item, ["private_test_cases", "private_tests", "private_testcases", "private"])
#         public_tcs = parse_testcases(public_tc_raw)
#         private_tcs = parse_testcases(private_tc_raw)

#         # "gold" solution (usually not present in LCB releases; we keep it nullable)
#         _, gold = first_present(item, ["reference_solution", "canonical_solution", "gold_solution", "solution"])
#         if gold is not None and not isinstance(gold, str):
#             gold = str(gold)

#         # LLM solutions (expect 10)
#         llm_solutions = llm_map.get(str(qid), [])
#         if len(llm_solutions) != 10:
#             # Keep going, but record what you actually have.
#             pass

#         sol_results = []
#         for s_i, sol in enumerate(llm_solutions):
#             try:
#                 r = evaluate_solution(item, sol, timeout_s=args.timeout_s)
#                 sol_results.append({
#                     "index": s_i,
#                     "code": sol,
#                     "passed": r.passed,
#                     "n_tests_run": r.n_tests_run,
#                     "n_tests_total": r.n_tests_total,
#                     "failure": r.failure,
#                 })
#             except Exception as e:
#                 sol_results.append({
#                     "index": s_i,
#                     "code": sol,
#                     "passed": False,
#                     "n_tests_run": 0,
#                     "n_tests_total": len(public_tcs) + len(private_tcs),
#                     "failure": f"Evaluator exception: {type(e).__name__}: {e}",
#                 })

#         out_records.append({
#             "question_id": qid,
#             "nl_description": desc,
#             "public_test_case_inputs": [tc.get("input", tc.get("stdin", tc.get("in"))) for tc in public_tcs],
#             "private_test_case_inputs": [tc.get("input", tc.get("stdin", tc.get("in"))) for tc in private_tcs],
#             "gold_code_solution": gold,  # likely null for official LCB datasets
#             "llm_solutions": sol_results,
#         })

#     with open(args.out, "w", encoding="utf-8") as f:
#         json.dump(out_records, f, ensure_ascii=False, indent=2)

#     print(f"Wrote {len(out_records)} tasks to {args.out}")


def load_lcb_index(args):
    # from datasets import load_dataset
    # ds = load_dataset(dataset_name, version_tag=version_tag)[split]
    benchmarks, _ = build_prompt_benchmark(args)
    idx = {}
    for benchmark in benchmarks:
        idx[benchmark.question_id] = benchmark
    return idx


def extract_solutions(task_obj):
    # we only want to look at tasks with at least one passing solution
    if not any(task_obj["graded_list"]):
        return None
    # let's get pairs of solutions and grades
    sols = []
    for sol, grade, meta in zip(task_obj["code_list"], task_obj["graded_list"], task_obj["metadata"]):
        sols.append((sol, grade, ast.literal_eval(meta)))
    # de-dup preserve order
    seen = set()
    out = []
    for s, g, m in sols:
        h = sha256_text(s)
        if h not in seen:
            seen.add(h)
            out.append((s, g, m))
    return out


def call_openai_code_mutator(
    client,
    positive: bool,  
    nl_prompt: str,
    seed_code: str,
    model
) -> str:
    """
    Returns Python code (string). Might include fences; caller should strip.
    """
    # Small randomness for style variation
    style_knobs = [
        "use helper functions and avoid global logic",
        "use a different input parsing strategy than the seed",
        "use a different control-flow structure than the seed (e.g., for vs while, iterators vs indexing)",
        "avoid copying any long contiguous block from the seed",
        "use a different data structure organization than the seed",
    ]
    random.shuffle(style_knobs)
    style = "; ".join(style_knobs[:3])

    if positive:
        system = (
            "You rewrite Python programs. Output ONLY valid Python code, no explanations, no markdown. "
            "The rewritten program must solve the given task correctly for all inputs."
        )
        user = f"""TASK (natural language):
{nl_prompt}

SEED SOLUTION (for reference only; do NOT copy large chunks verbatim):
{seed_code}

Rewrite the solution so it is substantially different in structure and style, while remaining correct.
Constraints:
- Output ONLY Python code.
- Must read from stdin and write to stdout.
- Must be meaningfully different (not simple renaming). 
- For example, {style}.
"""
    else:
        # Negative: runnable but wrong on some edge cases
        bug_types = [
            "off-by-one boundary mistake in a loop or index",
            "incorrect handling of zeros or special-case values",
            "wrong comparison operator in a critical condition",
            "incorrect tie-breaking choice (min/max) in one place",
            "incorrect initialization of an accumulator",
            "incorrectly skipping one element or one iteration",
        ]
        random.shuffle(bug_types)
        bugs = "; ".join(bug_types[:3])
        system = (
            "You write Python programs. Output ONLY valid Python code, no explanations, no markdown. "
            "Write a plausible but subtly incorrect solution that still runs without errors."
        )
        user = f"""TASK (natural language):
{nl_prompt}

SEED SOLUTION (reference; do NOT copy large chunks verbatim):
{seed_code}

Write a NEW Python solution that looks plausible and is syntactically valid, but is subtly incorrect.
It MUST:
- Read stdin and write stdout correctly (no crashes, no infinite loops).
- Be meaningfully different than the seed (not just renaming). For instance, {style}.
- For example, the solution may have one of the following types of bugs: {bugs}.
- Likely passes many basic cases but fails at least one edge case.

Output ONLY Python code.
"""

    # Use Responses API (recommended)
    resp = client.responses.create(
        model=model,
        input=[
            {"role": "system", "content": system},
            {"role": "user", "content": user},
        ],
        # You can tune these
        temperature=0.7,
        max_output_tokens=1200,
    )
    return resp.output_text


def robust_openai_call(*args, **kwargs) -> str:
    """
    Simple retry wrapper for transient errors / rate limits.
    """
    max_retries = kwargs.pop("_max_retries", 6)
    base_sleep = kwargs.pop("_base_sleep", 1.0)

    for k in range(max_retries):
        try:
            return call_openai_code_mutator(*args, **kwargs)
        except Exception as e:
            # backoff
            sleep = base_sleep * (2 ** k)
            if k == max_retries - 1:
                raise
            time.sleep(sleep)


# TODO: fuzz with LLM
def semantics_preserving_fuzz(code: str, rng: random.Random):
    pass


def semantic_break_fuzz(code: str, rng: random.Random):
    pass
    # tree, _ = ast_roundtrip(code)
    # if tree is None:
    #     return None, ["parse_failed"]

    # transforms = []

    # # First, a semantics-preserving pass to keep it “non-trivial”
    # mapping = make_renaming_mapping(tree, rng)
    # if mapping and rng.random() < 0.8:
    #     tree = RenameVariables(mapping).visit(tree)
    #     transforms.append(f"rename_vars({len(mapping)})")

    # tree = InsertDeadCode(rng).visit(tree)
    # transforms.append("insert_dead_code")

    # # Then inject ONE semantic bug (try in random order until one applies)
    # injectors = [
    #     ("range_off_by_one", MutateRangeOffByOne),
    #     ("flip_comparison", FlipComparison),
    #     ("swap_add_sub", SwapAddSub),
    # ]
    # rng.shuffle(injectors)

    # injected = False
    # for name, Cls in injectors:
    #     inj = Cls(rng).visit(ast.fix_missing_locations(tree))
    #     # Check if the injector actually mutated (via attribute)
    #     mutated = getattr(Cls(rng), "mutated", None)  # dummy instance: not useful
    #     # Instead: detect by re-running with same instance
    #     inst = Cls(rng)
    #     tree2 = inst.visit(tree)
    #     if getattr(inst, "mutated", False):
    #         tree = tree2
    #         transforms.append(name)
    #         injected = True
    #         break

    # if not injected:
    #     # fallback: still return a no-op-ish transform (won't help negatives much, but avoids crash)
    #     transforms.append("no_injector_applied")

    # ast.fix_missing_locations(tree)
    # try:
    #     return ast.unparse(tree), transforms
    # except Exception:
    #     return None, transforms + ["unparse_failed"]


def strip_code_fences(text: str) -> str:
    """
    Extract code from typical markdown fences. If no fences, return as-is.
    """
    t = text.strip()
    # ```python ... ```
    m = re.search(r"```(?:python)?\s*(.*?)\s*```", t, flags=re.DOTALL | re.IGNORECASE)
    if m:
        return m.group(1).strip()
    # fallback: remove leading/trailing triple backticks if present
    t = re.sub(r"^\s*```(?:python)?\s*", "", t, flags=re.IGNORECASE)
    t = re.sub(r"\s*```\s*$", "", t)
    return t.strip()


def run_python_program(code: str, stdin_text: str, timeout_s: float):
    with tempfile.TemporaryDirectory() as td:
        path = os.path.join(td, "main.py")
        with open(path, "w", encoding="utf-8") as f:
            f.write(code)
        try:
            proc = subprocess.run(
                [sys.executable, path],
                input=stdin_text.encode("utf-8"),
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                timeout=timeout_s,
            )
            return proc.returncode, proc.stdout.decode("utf-8", "replace"), proc.stderr.decode("utf-8", "replace")
        except subprocess.TimeoutExpired as e:
            out = (e.stdout or b"").decode("utf-8", "replace")
            err = (e.stderr or b"").decode("utf-8", "replace")
            return 124, out, err


def normalize_output(s: str) -> str:
    s = s.replace("\r\n", "\n")
    lines = [ln.rstrip() for ln in s.split("\n")]
    while lines and lines[-1] == "":
        lines.pop()
    return "\n".join(lines)


def evaluate_solution_stdin(code: str, tests, timeout_s: float):
    passed_mask = []
    failed_ids = []

    for test_id, tc in enumerate(tests):
        rc, out, err = run_python_program(code, tc.input, timeout_s)
        if rc == 124:
            return EvalResult(
                ok=False, passed_all=False, passed_mask=[], failed_test_ids=[], pass_fraction=0.0,
                error_kind="TLE", error_detail=err[:800]
            )
        if rc != 0:
            return EvalResult(
                ok=False, passed_all=False, passed_mask=[], failed_test_ids=[], pass_fraction=0.0,
                error_kind="RE", error_detail=err[:800]
            )
        good = (normalize_output(out) == normalize_output(tc.output))
        passed_mask.append(good)
        if not good:
            failed_ids.append(test_id)

    passed_all = all(passed_mask) if passed_mask else False
    frac = (sum(1 for b in passed_mask if b) / len(passed_mask)) if passed_mask else 0.0
    return EvalResult(
        ok=True, 
        passed_all=passed_all, 
        passed_mask=passed_mask,
        failed_test_ids=failed_ids, 
        pass_fraction=frac
    )


def curate_for_task(
    qid: str,
    solutions,
    tests,
    nl_prompt,
    client,
    model
):

    NUM_POSITIVE_SOLS = 5
    NUM_NEGATIVE_SOLS = 5

    # gold solutions pass all tests
    gold_solutions = [sol for sol, g, m in solutions if g]
    # non-trivial failing solutions fail some test, but isn't wrong in a trivial way
    nontrivial_failing_solutions = [sol for sol, g, m in solutions if not g and m["error_message"] != "Wrong answer: mismatched output length"]

    positive_solutions = []
    negative_solutions = []

    pos_codes = []
    neg_codes = []

    seen_hashes = set()

    attempts = 0
    while (len(positive_solutions) < NUM_POSITIVE_SOLS or len(negative_solutions) < NUM_NEGATIVE_SOLS) and attempts < 50:
        attempts += 1

        # Decide whether to try pos or neg this round (prioritize whichever is missing)
        need_pos = len(positive_solutions) < NUM_POSITIVE_SOLS
        need_neg = len(negative_solutions) < NUM_NEGATIVE_SOLS
        try_pos = need_pos and (not need_neg or random.random() < 0.55)

        if try_pos:
            seed_code = random.choice(gold_solutions)
        else:
            pool = gold_solutions + nontrivial_failing_solutions
            seed_code = random.choice(pool)
            

        # LLM call
        raw = robust_openai_call(
            positive=try_pos,
            nl_prompt=nl_prompt,
            seed_code=seed_code,
            client=client,
            model=model
        )

        code = strip_code_fences(raw)
        if not code:
            continue
        h = sha256_text(code)
        if h in seen_hashes:
            continue

        timeout_s = 5.0
        rr = evaluate_solution_stdin(code, tests, timeout_s)

        if try_pos:
            if rr.ok and rr.passed_all:
                seen_hashes.add(h)
                pos_codes.append(code)
                positive_solutions.append({
                    "code": code,
                })
        else:
            # Must RUN without errors, but FAIL at least one test
            if rr.ok and (not rr.passed_all) and rr.failed_test_ids:

                seen_hashes.add(h)
                neg_codes.append(code)

                negative_solutions.append({
                    "code": code,
                    "pass_fraction": rr.pass_fraction,
                    "failed_test_ids": rr.failed_test_ids,
                    # "passed_test_ids" : rr.passed_test_ids
                })


    all_failed_test_ids_lists = [item["failed_test_ids"] for item in negative_solutions]
    all_failed_test_ids = set([item for sublist in all_failed_test_ids_lists for item in sublist])

    failure_coverage = len(all_failed_test_ids)/len(tests)

    # all_passed_test_ids_lists = [item["passed_test_ids"] for item in negative_solutions]
    # all_passed_test_ids = set([item for sublist in all_passed_test_ids_lists for item in sublist])

    # pass_coverage = len(all_passed_test_ids)/len(tests)



    if len(positive_solutions) < NUM_POSITIVE_SOLS or len(negative_solutions) < NUM_NEGATIVE_SOLS:
        return None

    return {
        "question_id": qid,
        "n_generated_solutions": len(solutions),
        "n_tests": len(tests),
        "positives": positive_solutions,
        "negatives": negative_solutions,
        "attempts": attempts,
        "pass_fractions" : [item["pass_fraction"] for item in negative_solutions],
        "failure_coverage" : failure_coverage,
        # "pass_coverage" : pass_coverage
    }


def sha256_text(s: str) -> str:
    return hashlib.sha256(s.encode("utf-8")).hexdigest()


def write_task_fuzzed_files(qid: str, question_content: str, task_result: dict) -> None:
    """
    Writes fuzzed positives/negatives to python files for manual inspection.
    Expects task_result to have keys:
      - "positives": list of dicts with at least {"code": str}
      - "negatives": list of dicts with at least {"code": str, "pass_fraction": float, "failed_test_ids": list[str]}
    """

    def write_one(kind: str, idx: int, item: dict) -> None:
        code = item.get("code", "")
        if not isinstance(code, str):
            return


        header_lines = [
            f"# idx: {idx}",
            f"question: {question_content} ",
            f"kind: {kind}"
        ]

        if kind == "neg":
            pf = item.get("pass_fraction", None)
            failed = item.get("failed_test_ids", None)
            if pf is not None:
                header_lines.append(f"# pass_fraction: {pf}")
            if failed:
                header_lines.append(f"# failed_test_ids: {failed}")

        header = "'''\n" + "\n".join(header_lines) + "\n'''\n\n"

        path = f"{CODE_PATH}/{qid}_{kind}_{idx}.py"
        with open(path, "w", encoding="utf-8") as f:
            f.write(header + code.strip() + "\n")

    positives = task_result.get("positives", [])
    negatives = task_result.get("negatives", [])

    for i, it in enumerate(positives):
        write_one("pos", i, it)

    for i, it in enumerate(negatives):
        write_one("neg", i, it)


CODE_PATH = "output/code"

def extract_property_test_benchmarks2() -> None:
    # set up OpenAI client
    OPEN_AI_KEY_FILEPATH = "../open-ai-key.txt"
    with open(OPEN_AI_KEY_FILEPATH, 'r') as file:
        open_ai_key = file.read().strip()  # Reads the entire file content as a single string
        client = OpenAI(
            api_key=open_ai_key
        )

    # clear code directory
    
    if os.path.exists(CODE_PATH):
        shutil.rmtree(CODE_PATH) 
    os.makedirs(CODE_PATH) 


    args = get_args()

    random.seed(123)

    # input_tasks = load_input_tasks()
    model = LanguageModelStore[args.model]

    # load benchmarks with generated code
    print("loading benchmarks...")
    input_path = f"output/{model.model_repr}_property_test_benchmarks.json"
    with open(input_path, "r", encoding="utf-8") as f:
        input_tasks = json.load(f)


    # match benchmarks to their ids in the lcb dataset
    # we need this to access the test cases
    print("loading lcb dataset...")
    lcb_index = load_lcb_index(args)

    curated = []
    for obj in tqdm(input_tasks):
        qid = obj["question_id"]
        benchmark = lcb_index.get(qid)
        if benchmark is None:
            continue

        tests = benchmark.get_test_cases()
        if not tests:
            continue

        sols = extract_solutions(obj)
        if sols is None:
            continue

        nl_prompt = obj["question_content"]

        benchmark_res = curate_for_task(
            qid=qid,
            solutions=sols,
            tests=tests,
            nl_prompt=nl_prompt,
            client=client,
            model=args.model
        )

        if benchmark_res is None:
            continue

        curated.append(benchmark_res)
        write_task_fuzzed_files(qid, nl_prompt, benchmark_res)


    pass_rate_lists = [item["pass_fractions"] for item in curated]
    avg_pass_rate = statistics.mean([item for sublist in pass_rate_lists for item in sublist])
    avg_failure_coverage = statistics.mean([item["failure_coverage"] for item in curated])

    output = {
        "benchmarks" : curated,
        "avg_pass_rate" : avg_pass_rate, 
        "avg_failure_coverage" : avg_failure_coverage
    }

    output_path = f"output/{model.model_repr}_property_test_benchmarks2.json"

    if os.path.exists(output_path):
        os.remove(output_path)

    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(output, f, ensure_ascii=False, indent=2)

    print(f"Wrote {len(curated)} benchmarks to {output_path}")


def run_property_test_experiment():
    args = get_args()

    random.seed(123)

    model = LanguageModelStore[args.model]
    input_path = f"output/{model.model_repr}_property_test_benchmarks2.json"
    print("loading benchmarks...")
    input_path = f"output/{model.model_repr}_property_test_benchmarks.json"
    with open(input_path, "r", encoding="utf-8") as f:
        input_tasks = json.load(f)

    # for each benchmark:
    # 1. generate property-based tests from question_content using LLM (zero-shot)
    # 2. check if the property-based tests pass the positive solutions and fail the negative solutions
    # 3. get f1 score


if __name__ == "__main__":
    # main()
    extract_property_test_benchmarks()
    extract_property_test_benchmarks2()
    run_property_test_experiment()
