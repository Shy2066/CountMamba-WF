"""Run CountMamba experiments, scheduling independent tasks across GPUs.

The existing ``main.py`` and ``test.py`` remain the source of truth for model
training and evaluation. This file supplies explicit paths, schedules each
task on one GPU, and records commands/results in an isolated run directory.
"""

import argparse
import csv
from concurrent.futures import FIRST_COMPLETED, ThreadPoolExecutor, wait
import json
import os
from datetime import datetime
from pathlib import Path
import statistics
import subprocess
import sys


SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent
DATA_ROOT = PROJECT_ROOT / "npz_dataset"
RUNS_ROOT = PROJECT_ROOT / "logs" / "CountMamba_domain_runs"

DEFAULT_DATASETS = ("tor_fiber", "tor_starlink")
FIXED_ARGS = (
    "--seq_len", "5000",
    "--maximum_load_time", "120",
    "--max_matrix_len", "2700",
    "--maximum_cell_number", "3",
    "--log_transform",
)


def parse_args():
    parser = argparse.ArgumentParser(
        description="Run CountMamba F2F/F2S/S2S/S2F experiments across GPUs"
    )
    parser.add_argument(
        "--seeds", nargs="+", type=int, default=[2024],
        help="Training/evaluation seeds (default: 2024)",
    )
    parser.add_argument(
        "--datasets", nargs=2, action="append", default=None,
        metavar=("FIBER_DATASET", "STARLINK_DATASET"),
        help=(
            "paired dataset directory names under npz_dataset; repeat this "
            "option to run multiple defended pairs (default: tor_fiber "
            "tor_starlink)"
        ),
    )
    parser.add_argument(
        "--gpu", default="0",
        help=(
            "GPU IDs available to the runner, comma-separated; independent "
            "tasks run concurrently, one per GPU (default: 0)"
        ),
    )
    parser.add_argument(
        "--run-id", default=None,
        help="Isolated output directory name under logs/CountMamba_domain_runs",
    )
    parser.add_argument(
        "--dry-run", action="store_true",
        help="Print the planned 2-train/4-eval tasks per seed without executing",
    )
    return parser.parse_args()


def make_run_id(requested):
    if requested is None:
        return datetime.now().strftime("domain_%Y%m%d_%H%M%S_%f")
    if not requested or Path(requested).name != requested:
        raise ValueError("--run-id must be a non-empty directory name without path separators")
    return requested


def parse_gpu_ids(gpu_arg):
    gpu_ids = [gpu.strip() for gpu in gpu_arg.split(",")]
    if not gpu_ids or any(not gpu for gpu in gpu_ids):
        raise ValueError("--gpu must contain comma-separated, non-empty GPU IDs")
    if len(set(gpu_ids)) != len(gpu_ids):
        raise ValueError("--gpu must not contain duplicate GPU IDs")
    return gpu_ids


def make_directions(datasets):
    fiber, starlink = datasets
    return (
        ("F2F", fiber, fiber),
        ("F2S", fiber, starlink),
        ("S2S", starlink, starlink),
        ("S2F", starlink, fiber),
    )


def make_dataset_groups(dataset_args):
    dataset_pairs = dataset_args or [list(DEFAULT_DATASETS)]
    groups = []
    seen = set()
    for index, pair in enumerate(dataset_pairs, start=1):
        datasets = tuple(pair)
        if datasets in seen:
            raise ValueError(f"duplicate --datasets pair: {' '.join(datasets)}")
        seen.add(datasets)
        groups.append({
            "group_id": f"group_{index:02d}",
            "datasets": datasets,
            "directions": make_directions(datasets),
        })
    return groups


def command_string(command):
    return " ".join(subprocess.list2cmdline([part]) for part in command)


def write_summary(summary, run_dir):
    summary_path = run_dir / "summary.json"
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    with summary_path.open("w", encoding="utf-8") as handle:
        json.dump(summary, handle, indent=2, sort_keys=True)

    csv_path = run_dir / "summary.csv"
    fields = [
        "group_id", "seed", "direction", "source_dataset", "target_dataset",
        "accuracy", "f1", "checkpoint_file", "result_file", "log_file",
        "gpu", "status", "error",
    ]
    with csv_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for record in summary["evaluations"]:
            writer.writerow({field: record.get(field, "") for field in fields})


def make_summary(run_id, seeds, gpu_ids, groups):
    evaluations = []
    for group in groups:
        for seed in seeds:
            for direction, source, target in group["directions"]:
                evaluations.append({
                    "group_id": group["group_id"],
                    "seed": seed,
                    "direction": direction,
                    "source_dataset": source,
                    "target_dataset": target,
                    "accuracy": None,
                    "f1": None,
                    "checkpoint_file": "",
                    "result_file": "",
                    "log_file": "",
                    "status": "pending",
                    "error": "",
                })
    first_datasets = groups[0]["datasets"]
    return {
        "run_id": run_id,
        "status": "running",
        "gpu": ",".join(gpu_ids),
        "gpus": gpu_ids,
        "seeds": seeds,
        "datasets": {
            "fiber": first_datasets[0],
            "starlink": first_datasets[1],
        },
        "dataset_groups": [
            {
                "group_id": group["group_id"],
                "fiber": group["datasets"][0],
                "starlink": group["datasets"][1],
            }
            for group in groups
        ],
        "config": {
            "seq_len": 5000,
            "maximum_load_time": 120,
            "max_matrix_len": 2700,
            "maximum_cell_number": 3,
            "log_transform": True,
            "split": "train.npz/valid.npz/test.npz (4860/540/600)",
        },
        "training": [],
        "evaluations": evaluations,
        "aggregate": {},
        "aggregate_by_group": {},
    }


def build_train_command(python, dataset, seed, checkpoint):
    return [
        python, "main.py", "--dataset", dataset, "--seed", str(seed),
        "--checkpoint-file", str(checkpoint), *FIXED_ARGS,
    ]


def build_eval_command(python, source_dataset, target_dataset, seed, checkpoint, output):
    # source_dataset is intentionally accepted here to make the call site
    # self-documenting; the target dataset is the test.py dataset argument.
    del source_dataset
    return [
        python, "test.py", "--dataset", target_dataset, "--seed", str(seed),
        "--checkpoint-file", str(checkpoint), "--output-file", str(output),
        *FIXED_ARGS,
    ]


def tail_text(path, lines=30):
    try:
        content = path.read_text(encoding="utf-8", errors="replace").splitlines()
    except OSError as exc:
        return f"could not read log: {exc}"
    return "\n".join(content[-lines:])


def run_command(command, log_file, env):
    log_file.parent.mkdir(parents=True, exist_ok=True)
    with log_file.open("w", encoding="utf-8") as handle:
        handle.write(f"$ {command_string(command)}\n\n")
        handle.flush()
        try:
            completed = subprocess.run(
                command, cwd=SCRIPT_DIR, env=env, stdout=handle,
                stderr=subprocess.STDOUT, check=False,
            )
        except OSError as exc:
            handle.write(f"\n[runner error] {exc}\n")
            return 127, str(exc)
    if completed.returncode != 0:
        return completed.returncode, tail_text(log_file)
    return 0, ""


def update_aggregate(summary, groups):
    aggregate_by_group = {}
    for group in groups:
        group_aggregate = {}
        for direction, _, _ in group["directions"]:
            records = [
                record for record in summary["evaluations"]
                if record["group_id"] == group["group_id"]
                and record["direction"] == direction
                and record["status"] == "success"
            ]
            if not records:
                continue
            values = {
                "accuracy": [record["accuracy"] for record in records],
                "f1": [record["f1"] for record in records],
            }
            group_aggregate[direction] = {}
            for metric, numbers in values.items():
                group_aggregate[direction]["mean_" + metric] = round(
                    statistics.mean(numbers), 4
                )
                group_aggregate[direction]["std_" + metric] = round(
                    statistics.stdev(numbers), 4
                ) if len(numbers) > 1 else 0.0
        aggregate_by_group[group["group_id"]] = group_aggregate
    summary["aggregate_by_group"] = aggregate_by_group
    summary["aggregate"] = (
        aggregate_by_group[groups[0]["group_id"]] if len(groups) == 1 else {}
    )


def validate_inputs(groups):
    datasets = []
    for group in groups:
        datasets.extend(group["datasets"])
    for dataset in datasets:
        if not dataset or Path(dataset).name != dataset:
            raise ValueError(
                "--datasets values must be directory names without path separators"
            )

    missing = []
    for dataset in dict.fromkeys(datasets):
        for split in ("train.npz", "valid.npz", "test.npz"):
            path = DATA_ROOT / dataset / split
            if not path.is_file():
                missing.append(str(path))
    if missing:
        raise FileNotFoundError("Missing required split files:\n" + "\n".join(missing))


def dry_run(python, seeds, gpu, groups):
    train_count = 0
    eval_count = 0
    gpu_ids = parse_gpu_ids(gpu)
    print(
        f"DRY-RUN: {len(groups)} dataset group(s), {len(seeds)} seed(s), "
        "2 training and 4 evaluation tasks per group/seed"
    )
    print(
        f"DRY-RUN GPUs: {', '.join(gpu_ids)} "
        f"({len(gpu_ids)} concurrent task(s) maximum)"
    )
    for group in groups:
        datasets = group["datasets"]
        directions = group["directions"]
        group_root = Path("<run_dir>")
        if len(groups) > 1:
            group_root = group_root / "groups" / group["group_id"]
        print(
            f"{group['group_id']}: fiber={datasets[0]}, "
            f"starlink={datasets[1]}"
        )
        for seed in seeds:
            print(f"  seed={seed}")
            for dataset in datasets:
                checkpoint = group_root / "checkpoints" / f"{dataset}_seed_{seed}.pth"
                print("    TRAIN", command_string(build_train_command(python, dataset, seed, checkpoint)))
                train_count += 1
            for direction, source, target in directions:
                checkpoint = group_root / "checkpoints" / f"{source}_seed_{seed}.pth"
                output = group_root / "results" / f"{direction}_seed_{seed}.json"
                print("    EVAL ", command_string(build_eval_command(python, source, target, seed, checkpoint, output)))
                eval_count += 1
    print(f"DRY-RUN COUNTS: train={train_count}, eval={eval_count}")
    expected = len(groups) * len(seeds)
    return 0 if train_count == 2 * expected and eval_count == 4 * expected else 1


def execute(args):
    gpu_ids = parse_gpu_ids(args.gpu)
    groups = make_dataset_groups(args.datasets)
    validate_inputs(groups)
    run_id = make_run_id(args.run_id)
    run_dir = RUNS_ROOT / run_id
    if run_dir.exists():
        raise FileExistsError(f"Refusing to overwrite existing run directory: {run_dir}")
    run_dir.mkdir(parents=True)

    python = sys.executable
    summary = make_summary(run_id, args.seeds, gpu_ids, groups)
    write_summary(summary, run_dir)

    tasks = []
    train_task_ids = {}
    next_task_id = 0
    for group in groups:
        group_id = group["group_id"]
        datasets = group["datasets"]
        directions = group["directions"]
        group_dir = run_dir if len(groups) == 1 else run_dir / "groups" / group_id
        for seed in args.seeds:
            for dataset in datasets:
                checkpoint = group_dir / "checkpoints" / f"{dataset}_seed_{seed}.pth"
                log_file = group_dir / "logs" / f"train_{dataset}_seed_{seed}.log"
                training = {
                    "group_id": group_id,
                    "seed": seed,
                    "dataset": dataset,
                    "checkpoint_file": str(checkpoint),
                    "log_file": str(log_file),
                    "status": "pending",
                }
                summary["training"].append(training)
                task_id = next_task_id
                next_task_id += 1
                previous_task_id = train_task_ids.get((group_id, seed, dataset))
                train_task_ids[(group_id, seed, dataset)] = task_id
                task = {
                    "id": task_id,
                    "kind": "train",
                    "record": training,
                    "dataset": dataset,
                    "seed": seed,
                    "checkpoint": checkpoint,
                    "log_file": log_file,
                    "group_dir": group_dir,
                    "group_id": group_id,
                }
                if previous_task_id is not None:
                    task["dependency"] = previous_task_id
                tasks.append(task)

    for group in groups:
        group_id = group["group_id"]
        directions = group["directions"]
        group_dir = run_dir if len(groups) == 1 else run_dir / "groups" / group_id
        for seed in args.seeds:
            for direction, source, target in directions:
                record = next(
                    item for item in summary["evaluations"]
                    if item["group_id"] == group_id
                    and item["seed"] == seed
                    and item["direction"] == direction
                )
                checkpoint = group_dir / "checkpoints" / f"{source}_seed_{seed}.pth"
                output = group_dir / "results" / f"{direction}_seed_{seed}.json"
                log_file = group_dir / "logs" / f"eval_{direction}_seed_{seed}.log"
                task_id = next_task_id
                next_task_id += 1
                tasks.append({
                    "id": task_id,
                    "kind": "eval",
                    "record": record,
                    "direction": direction,
                    "source": source,
                    "target": target,
                    "seed": seed,
                    "checkpoint": checkpoint,
                    "output": output,
                    "log_file": log_file,
                    "group_dir": group_dir,
                    "group_id": group_id,
                    "dependency": train_task_ids[(group_id, seed, source)],
                })

    pending_tasks = list(tasks)
    task_states = {task["id"]: "pending" for task in tasks}
    running = {}
    available_gpus = list(gpu_ids)
    failed = False

    with ThreadPoolExecutor(max_workers=len(gpu_ids)) as executor:
        while pending_tasks or running:
            while pending_tasks and available_gpus:
                task_index = next(
                    (
                        index for index, task in enumerate(pending_tasks)
                        if task.get("dependency") is None
                        or task_states[task["dependency"]] in ("success", "failed", "skipped")
                    ),
                    None,
                )
                if task_index is None:
                    break
                task = pending_tasks.pop(task_index)
                dependency = task.get("dependency")
                if dependency is not None and task_states[dependency] != "success":
                    task_states[task["id"]] = "skipped"
                    task["record"]["status"] = "skipped"
                    task["record"]["error"] = "not run because a prerequisite task failed"
                    write_summary(summary, run_dir)
                    continue

                gpu = available_gpus.pop(0)
                task["record"]["status"] = "running"
                task["record"]["gpu"] = gpu
                if task["kind"] == "eval":
                    task["record"].update({
                        "checkpoint_file": str(task["checkpoint"]),
                        "result_file": str(task["output"]),
                        "log_file": str(task["log_file"]),
                    })
                    command = build_eval_command(
                        python, task["source"], task["target"], task["seed"],
                        task["checkpoint"], task["output"],
                    )
                    label = (
                        f"EVAL GPU={gpu} group={task['group_id']} "
                        f"{task['direction']} seed={task['seed']}"
                    )
                else:
                    command = build_train_command(
                        python, task["dataset"], task["seed"], task["checkpoint"]
                    )
                    label = (
                        f"TRAIN GPU={gpu} group={task['group_id']} "
                        f"seed={task['seed']} dataset={task['dataset']}"
                    )
                env = os.environ.copy()
                env["CUDA_VISIBLE_DEVICES"] = gpu
                task_states[task["id"]] = "running"
                print(f"{label}: {task['log_file']}", flush=True)
                write_summary(summary, run_dir)
                future = executor.submit(
                    run_command, command, task["log_file"], env
                )
                running[future] = (task, gpu)

            if not running:
                if pending_tasks:
                    raise RuntimeError("Experiment task dependencies could not be resolved")
                break

            completed, _ = wait(running, return_when=FIRST_COMPLETED)
            for future in completed:
                task, gpu = running.pop(future)
                available_gpus.append(gpu)
                available_gpus.sort(key=gpu_ids.index)
                try:
                    returncode, error = future.result()
                except Exception as exc:
                    returncode, error = 1, str(exc)

                record = task["record"]
                if task["kind"] == "train":
                    if returncode == 0 and task["checkpoint"].is_file():
                        task_states[task["id"]] = "success"
                        record["status"] = "success"
                        record["returncode"] = 0
                    else:
                        task_states[task["id"]] = "failed"
                        record["status"] = "failed"
                        record["returncode"] = returncode
                        record["error"] = error or (
                            "training exited successfully but produced no checkpoint"
                        )
                        failed = True
                        print(f"FAILED: {record['error']}", flush=True)
                else:
                    if returncode == 0 and task["output"].is_file():
                        try:
                            with task["output"].open(encoding="utf-8") as handle:
                                result = json.load(handle)
                            record["accuracy"] = result.get("Accuracy")
                            record["f1"] = result.get("F1-score")
                            if record["accuracy"] is None or record["f1"] is None:
                                raise ValueError(
                                    f"result lacks Accuracy/F1-score: {result}"
                                )
                        except (OSError, ValueError, json.JSONDecodeError) as exc:
                            returncode, error = 1, str(exc)
                    elif returncode == 0:
                        returncode, error = 1, (
                            "evaluation exited successfully but produced no JSON"
                        )
                    if returncode == 0:
                        task_states[task["id"]] = "success"
                        record["status"] = "success"
                    else:
                        task_states[task["id"]] = "failed"
                        record["status"] = "failed"
                        record["error"] = error
                        failed = True
                        print(f"FAILED: {record['error']}", flush=True)
                write_summary(summary, run_dir)

    summary["status"] = "failed" if failed else "complete"
    update_aggregate(summary, groups)
    write_summary(summary, run_dir)
    print(f"{'FAILED' if failed else 'COMPLETE'}: {run_dir}", flush=True)
    return 1 if failed else 0


def main():
    args = parse_args()
    if args.dry_run:
        groups = make_dataset_groups(args.datasets)
        validate_inputs(groups)
        return dry_run(sys.executable, args.seeds, args.gpu, groups)
    try:
        return execute(args)
    except Exception as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
