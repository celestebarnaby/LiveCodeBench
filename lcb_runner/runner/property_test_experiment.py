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
import csv
from pathlib import Path

from tqdm import tqdm
from openai import OpenAI
from dataclasses import dataclass
from sklearn.metrics import f1_score
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


def get_code_solutions_for_benchmarks(client):
    '''
    This script enumerates LCB benchmarks and generates code solutions via LLM prompting.
    The generated solutions are then saved. 
    '''
    args = get_args()

    model = LanguageModelStore[args.model]
    benchmarks, format_prompt = build_prompt_benchmark(args)

    benchmarks = benchmarks[:10]
    # random.shuffle(benchmarks)
    # NUM_BENCHMARKS = 10
    # random.seed(123)
    # benchmarks = random.sample(benchmarks, NUM_BENCHMARKS)

    output_path = f"output/{model.model_repr}_property_test_benchmarks.json"
    if os.path.exists(output_path):
        os.remove(output_path)

    # ---- Generate parse_input for each benchmark ----
    print("generating parse_input functions...")
    # parse_inputs_by_qid = {}
    # parse_errors_by_qid = {}
    parse_inputs_list = []

    for instance in tqdm(benchmarks):
        qid = instance.question_id
        # try:
        parse_code = get_parse_input(instance.question_content, client, args.model)
        parse_inputs_list.append(parse_code)

        # also write to a file for inspection
        # parse_path = f"output/parse_input/{qid}_parse_input.py"
        # with open(parse_path, "w", encoding="utf-8") as f:
        #     f.write(f"'''\n{instance.question_content}\n'''\n\n")
        #     f.write(parse_code)
        # except Exception as e:
        #     parse_errors_by_qid[qid] = str(e)
        #     parse_inputs_list.append("")

    # ---- Existing generation pipeline for code solutions ----

    print("get results")
    runner = build_runner(args, model)

    results: list[list[str]] = runner.run_main(benchmarks, format_prompt, parse_inputs_list)

    print("combine results")
    combined_results = combine_results(
        args.scenario, results, parse_inputs_list, model, args.cot_code_execution
    )

    print("save results")
    save_results = [
        instance.insert_output(outputs_list, extracted_list)
        for instance, (outputs_list, extracted_list) in zip(
            benchmarks, combined_results
        )
    ]

    save_results, combined_results = sort_and_extract_save_results(
        args.scenario, save_results
    )

    print("get metrics")
    metrics = get_metrics(args.scenario, args, benchmarks, combined_results)
    graded = extract_instance_results(metrics[1])

    metadatas = metrics[2]

    for instance, (_, extracted_list), graded_list, meta in zip(benchmarks, combined_results, graded, metadatas):
        for i, (extracted_code, code_grade, code_meta) in enumerate(zip(extracted_list, graded_list, meta)):
            path = f"output/generated_code/{instance.question_id}_{i}_code.py"
            with open(path, "w") as f:
                f.write(f"'''\n{instance.question_content}\n\npassed: {code_grade}\n\nmetadata: {code_meta}\n'''\n\n")
                f.write(extracted_code)

    save_eval_results = [
        instance.insert_output_evaluation(
            outputs_list, extracted_list, graded_list, metadata=meta
        )
        for instance, (outputs_list, extracted_list), graded_list, meta in zip(
            benchmarks, combined_results, graded, metadatas
        )
    ]

    # save_eval_results = [item for item in save_eval_results if item["pass@1"] > 0]
    # print(f"Saved results for {len(save_eval_results)} benchmarks")

    with open(output_path, "w") as f:
        json.dump(save_eval_results, f, indent=4)


def get_parse_input(task_description, client, model_name):
    prompt = f"""\
You are an expert Python programmer.

Given the following competitive programming problem statement, write Python code that defines:

- def parse_input(stdin: str) -> list[tuple]:
    * stdin is the full contents of standard input as a single string.
    * Return a list of test cases.
    * Each test case MUST be represented as a tuple containing only JSON-serializable primitives
      (int, str, bool) and lists of those primitives (no custom classes).
    * If the input has t test cases, the returned list MUST have length t.
    * Do NOT read from stdin, do NOT write to stdout.
    * No top-level side effects.

Problem statement:
{task_description}

Return ONLY valid Python code. Do NOT return any code other than the parse_input implementation. Do not include explanations.
"""

    response = client.chat.completions.create(
        model=model_name,
        messages=[
            {"role": "user", "content": prompt},
        ],
        temperature=0,
        max_completion_tokens=800,  # parsers can be a bit longer than 512
    )
    return strip_code_fences(response.choices[0].message.content)


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

Rewrite the `solve_task` function so it is substantially different in structure and style, while remaining correct.
Constraints:
- Output ONLY Python code.
- ONLY modify the `solve_task` function.
- Include the rest of the seed solution unchanged in your output.
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

Change the `solve_task` function so that it looks plausible and is syntactically valid, but is subtly incorrect.
It MUST:
- Be meaningfully different than the seed (not just renaming). For instance, {style}.
- For example, the solution may have one of the following types of bugs: {bugs}.
- Likely passes many basic cases but fails at least one edge case.
Contraints:
- Output ONLY Python code.
- ONLY modify the `solve_task` function.
- Include the rest of the seed solution unchanged in your output.
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
OPEN_AI_KEY_FILEPATH = "../open-ai-key.txt"

def generate_property_test_benchmarks() -> None:
    '''
    This script takes the generated code solutions for the benchmarks,
    and fuzzes them to generate a set of positive and negative solutions.
    These are our benchmarks for our property-based tests.
    '''

    # set up OpenAI client
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

    if len(curated) == 0:
        print("No benchmarks generated....")
        raise TypeError

    # each negative solution has its own pass rate
    pass_rate_lists = [item["pass_fractions"] for item in curated]
    avg_pass_rate = statistics.mean([item for sublist in pass_rate_lists for item in sublist])
    # each benchmark has its own failure coverage (across all negative solutions)
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
    print(f"Average unit test pass rate (higher is better): {avg_pass_rate}")
    print(f"Average unit test failure coverage (higher is better): {avg_failure_coverage}")


def generate_property_tests(client, prompt: str, model_name: str) -> str:
    """
    Generate Hypothesis-based property tests for the function `entry_point`
    described by the HumanEval prompt.
    """
    user_msg = f"""
Here is a description of a Python task:

\"\"\"{prompt}\"\"\"

The core logic of this task has been implemented in a function called `solve_task`.
This function has the following properties:
- Does NOT read from stdin, and does NOT write to stdout.
- Has NO top-level side effects.
- The input to the function is a single test case
- Returns the exact output.

Write a set of **Hypothesis property-based tests** for this function.

Requirements:
- Use `import hypothesis`. Do NOT import any other libraries.
- Define a single function named `check`, and then call that function.
- Call the function `solve_task` in your tests.
- Do NOT redefine the function itself.
- Only output Python test code, no explanations.
"""

    response = client.chat.completions.create(
        model=model_name,
        messages=[
            {
                "role": "system",
                "content": (
                    "You are an expert in property-based testing with Hypothesis. "
                    "You write concise and correct Hypothesis tests."
                ),
            },
            {"role": "user", "content": user_msg},
        ],
        # temperature=0.5,  # a bit more creativity for tests
        max_completion_tokens=512,
    )
    return strip_code_fences(response.choices[0].message.content)


ENTRY_POINT = "solve_task"

def _make_property_test_harness(
    solution_code: str,
    property_tests_code: str,
) -> str:
    """
    Combines:
      - candidate solution (must define solve_io(stdin)->str)
      - generated property tests (must define run_property_tests(solve_io)->bool OR raise AssertionError)
    into an executable script.
    """

    return f"""\
# --- Candidate solution ---
{solution_code}

# --- Generated property tests ---
{property_tests_code}

def __main():
    # Support two conventions:
    # (1) run_property_tests(solve_io) -> bool
    # (2) run_property_tests(solve_io) does assertions and returns None
    fn = {ENTRY_POINT}
    try:
        res = run_property_tests(fn)
        if res is False:
            raise AssertionError("run_property_tests returned False")
        print("PASS")
    except AssertionError as e:
        print("FAIL")
        raise
    except Exception as e:
        # Treat exceptions as failures (including generation errors)
        print("FAIL")
        raise

if __name__ == "__main__":
    __main()
"""



def solution_passes_property_tests(
    solution_code: str,
    property_tests_code: str,

) -> bool:
    harness = _make_property_test_harness(solution_code, property_tests_code)
    rc, out, err = _run_python(harness)
    return rc == 0


def _run_python(code: str) -> Tuple[int, str, str]:
    """Run python code in a temp file. Returns (returncode, stdout, stderr)."""
    with tempfile.TemporaryDirectory() as td:
        path = Path(td) / "prog.py"
        path.write_text(code, encoding="utf-8")
        try:
            p = subprocess.run(
                [sys.executable, str(path)],
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                timeout=6.0,
            )
            return p.returncode, p.stdout.decode("utf-8", "replace"), p.stderr.decode("utf-8", "replace")
        except subprocess.TimeoutExpired as e:
            out = (e.stdout or b"").decode("utf-8", "replace")
            err = (e.stderr or b"").decode("utf-8", "replace")
            return 124, out, err


def run_property_test_experiment():

    # set up OpenAI client
    with open(OPEN_AI_KEY_FILEPATH, 'r') as file:
        open_ai_key = file.read().strip()  # Reads the entire file content as a single string
        client = OpenAI(
            api_key=open_ai_key
        )

    args = get_args()

    random.seed(123)

    model = LanguageModelStore[args.model]
    input_path = f"output/{model.model_repr}_property_test_benchmarks2.json"
    print("loading benchmarks...")
    input_path = f"output/{model.model_repr}_property_test_benchmarks.json"
    with open(input_path, "r", encoding="utf-8") as f:
        input_tasks = json.load(f)


    model_to_f1_score = {}
    models =  [
        "gpt-5-mini",
        # "gpt-5.2",
        # "gpt-4.1"
    ]

    for model_name in models:
        y_true = []
        y_pred = []
        for task in input_tasks:
            prompt = task["question_content"]
            # generate property-based tests from question_content using LLM (zero-shot)
            property_tests_code = generate_property_tests(client, prompt, model_name)
            y_true += [1 for _ in task["positives"]] 
            y_true += [0 for _ in task["negatives"]]
            y_pred = []
            for sol_code in task["positives"] + task["negatives"]:
                ok = solution_passes_property_tests(sol_code, property_tests_code)
                y_pred.append(1 if ok else 0) 

        score = f1_score(y_true, y_pred, average='binary')
        model_to_f1_score[model_name] = score

    # write results to_csv
    results_file = "./output/property_test_experiment_results.csv"
    with open(results_file, mode='w', newline='') as file:
        writer = csv.writer(file)
        # Write all rows at once
        writer.writerow(["Model Name", "Property-Based Test F1 Score"])
        writer.writerows(list(model_to_f1_score))


if __name__ == "__main__":
    # main()

    with open(OPEN_AI_KEY_FILEPATH, 'r') as file:
        open_ai_key = file.read().strip()  # Reads the entire file content as a single string
        client = OpenAI(
            api_key=open_ai_key
        )

    get_code_solutions_for_benchmarks(client)
    generate_property_test_benchmarks()
    # run_property_test_experiment()
