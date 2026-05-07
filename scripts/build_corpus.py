"""
Encode cached corpus leads into FAISS-compatible embedding shards.

Runtime configuration is read from the repository-level "config.toml". This script reads "[corpus].lead_jsonl" under "[corpus].dataset_build_dir", uses "[model]" settings for the Harrier encoder, and writes shard/log outputs under the configured "[build]" paths. "[build].cuda_devices" controls CUDA device assignment. The parent process starts one worker per configured device, assigns each worker a single "CUDA_VISIBLE_DEVICES" value, and coordinates work with the internal "ENCODE_RANK" and "ENCODE_WORLD_SIZE" environment variables. If no CUDA devices are configured, the script falls back to the visible CUDA device count, or one CPU-visible worker when CUDA is unavailable. "download_datasets.py" is responsible for downloading actual data, this file only reads the downloaded data files and writes/caches "corpus.<rank>.pkl" shards for retrieval.
"""

import glob
import json
import os
import sys
import time
import pickle
import subprocess
import numpy as np
import tomllib

from pathlib import Path
from tqdm import tqdm
from types import SimpleNamespace
from typing import IO, List, Optional, Tuple

DEFAULT_MODEL = "microsoft/harrier-oss-v1-0.6b"
DEFAULT_SHARD_DIR = "indexes/harrier-oss-v1-0.6b"


def _repo_root() -> str:
    return os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def _load_config(config_path: str) -> dict:
    with open(config_path, "rb") as f:
        return tomllib.load(f)


def _data_path(config: dict, key: str, default: str) -> str:
    data_dir = config["corpus"]["dataset_build_dir"]
    path = config["corpus"].get(key, default)
    if os.path.isabs(path):
        return path
    return os.path.join(data_dir, path)


def _build_path(config: dict, key: str, default: str) -> str:
    data_dir = config["corpus"]["dataset_build_dir"]
    path = config.get("build", {}).get(key, default)
    if os.path.isabs(path):
        return path
    return os.path.join(data_dir, path)


def _settings(config: dict) -> SimpleNamespace:
    model_config = config.get("model", {})
    build_config = config.get("build", {})
    cuda_devices = build_config.get("cuda_devices")
    if cuda_devices:
        cuda_devices = [str(device) for device in cuda_devices]
    return SimpleNamespace(
        lead_jsonl=_data_path(config, "lead_jsonl", "leads.jsonl"),
        shard_dir=_build_path(config, "shard_dir", DEFAULT_SHARD_DIR),
        log_dir=_build_path(config, "log_dir", "logs"),
        model=model_config.get("name", DEFAULT_MODEL),
        model_cache_dir=model_config.get("cache_dir"),
        max_length=int(model_config.get("max_length", 1024)),
        batch_size=int(model_config.get("batch_size", 512)),
        dtype=model_config.get("dtype", "float16"),
        progress_interval=float(build_config.get("progress_interval", 5.0)),
        cuda_devices=cuda_devices,
        stream_logs=bool(build_config.get("stream_logs", False)),
    )


def _load_lead_slice(
    lead_jsonl: str, rank: int, world_size: int
) -> Tuple[List[str], List[str]]:
    ids, texts = [], []
    with open(lead_jsonl) as f:
        for line_idx, line in enumerate(f):
            if line_idx % world_size != rank:
                continue
            row = json.loads(line)
            ids.append(row["id"])
            texts.append(row["text"])
    return ids, texts


def step_encode(args: SimpleNamespace) -> None:
    import torch

    # Allow `python scripts/build_corpus.py ...` from repo root.
    sys.path.insert(0, _repo_root())
    from retrievers.encoder import HarrierEncoder

    if args.world_size < 1 or not (0 <= args.rank < args.world_size):
        raise ValueError("rank must satisfy 0 <= rank < world_size")

    os.makedirs(args.shard_dir, exist_ok=True)
    out_path = os.path.join(args.shard_dir, f"corpus.{args.rank:03d}.pkl")

    ids, texts = _load_lead_slice(args.lead_jsonl, args.rank, args.world_size)
    if not ids:
        print(f"[rank {args.rank}] no rows for this slice; skipping")
        return
    total_batches = (len(texts) + args.batch_size - 1) // args.batch_size
    print(
        f"[rank {args.rank}] encoding {len(texts):,} docs "
        f"in {total_batches:,} batches of up to {args.batch_size}",
        flush=True,
    )

    dtype = {"float16": torch.float16, "bfloat16": torch.bfloat16, "float32": torch.float32}[
        args.dtype
    ]
    encoder = HarrierEncoder(
        model_name=args.model,
        torch_dtype=dtype,
        max_length=args.max_length,
        cache_dir=args.model_cache_dir,
    )

    all_reps = []
    started = time.monotonic()
    progress = tqdm(
        total=len(texts),
        desc=f"Encode rank {args.rank}/{args.world_size}",
        unit="doc",
        unit_scale=True,
        dynamic_ncols=True,
        mininterval=args.progress_interval,
    )
    try:
        for start in range(0, len(texts), args.batch_size):
            batch = texts[start : start + args.batch_size]
            all_reps.append(encoder.encode_passages(batch))
            progress.update(len(batch))
            elapsed = max(time.monotonic() - started, 1e-9)
            docs_per_sec = progress.n / elapsed
            progress.set_postfix(
                batch=args.batch_size,
                docs_per_sec=f"{docs_per_sec:,.1f}",
            )
    finally:
        progress.close()
    reps = np.vstack(all_reps).astype(np.float32)

    with open(out_path, "wb") as f:
        pickle.dump((reps, ids), f)
    elapsed = max(time.monotonic() - started, 1e-9)
    print(
        f"[rank {args.rank}] wrote {reps.shape[0]:,} vectors of dim {reps.shape[1]} "
        f"-> {out_path} ({reps.shape[0] / elapsed:,.1f} docs/s)",
        flush=True,
    )


def _stream_logs(logs: List[str]) -> subprocess.Popen:
    return subprocess.Popen(["tail", "-n", "+1", "-F", *logs])


def _terminate(processes: List[subprocess.Popen]) -> None:
    for proc in processes:
        if proc.poll() is None:
            proc.terminate()
    for proc in processes:
        if proc.poll() is None:
            try:
                proc.wait(timeout=10)
            except subprocess.TimeoutExpired:
                proc.kill()
                proc.wait()


def encode_all(args: SimpleNamespace, config_path: str) -> None:
    if not os.path.exists(args.lead_jsonl):
        raise FileNotFoundError(f"Missing leads file: {args.lead_jsonl}")

    cuda_devices = args.cuda_devices
    if cuda_devices:
        num_gpus = len(cuda_devices)
    else:
        import torch

        num_gpus = torch.cuda.device_count() or 1
        cuda_devices = [str(rank) for rank in range(num_gpus)]

    if num_gpus < 1:
        raise ValueError("num_gpus must be at least 1")

    os.makedirs(args.shard_dir, exist_ok=True)
    os.makedirs(args.log_dir, exist_ok=True)

    print(f"Encoding {args.lead_jsonl}")
    print(f"Writing shards to {args.shard_dir}")
    print(f"Using {num_gpus} worker(s)")
    print(f"CUDA devices: {', '.join(cuda_devices)}")

    if num_gpus == 1:
        os.environ["CUDA_VISIBLE_DEVICES"] = cuda_devices[0]
        args.rank = 0
        args.world_size = 1
        step_encode(args)
        return

    script = Path(__file__).resolve()
    workers: List[subprocess.Popen] = []
    log_files: List[IO[str]] = []
    logs: List[str] = []
    tail_proc: Optional[subprocess.Popen] = None

    try:
        for rank in range(num_gpus):
            log_path = os.path.join(args.log_dir, f"encode.rank{rank}.log")
            log = open(log_path, "w")
            log_files.append(log)
            logs.append(log_path)
            cuda_device = cuda_devices[rank]
            print(f"  rank {rank} -> CUDA device {cuda_device} -> {log_path}")

            env = os.environ.copy()
            env["CUDA_VISIBLE_DEVICES"] = cuda_device
            env["ENCODE_RANK"] = str(rank)
            env["ENCODE_WORLD_SIZE"] = str(num_gpus)
            env["PYTHONUNBUFFERED"] = "1"
            cmd = [sys.executable, str(script)]
            workers.append(
                subprocess.Popen(cmd, stdout=log, stderr=subprocess.STDOUT, env=env)
            )

        if args.stream_logs:
            tail_proc = _stream_logs(logs)

        failed = False
        for proc in workers:
            if proc.wait() != 0:
                failed = True
                _terminate(workers)
                break
    except KeyboardInterrupt:
        _terminate(workers)
        raise
    finally:
        if tail_proc is not None:
            tail_proc.terminate()
            try:
                tail_proc.wait(timeout=5)
            except subprocess.TimeoutExpired:
                tail_proc.kill()
                tail_proc.wait()
        for log in log_files:
            log.close()

    if failed:
        raise RuntimeError(f"one or more encode workers failed; see {args.log_dir}/")

    print("[done] shards written:")
    for path in sorted(glob.glob(os.path.join(args.shard_dir, "corpus.*.pkl"))):
        size = os.path.getsize(path) / (1024 * 1024)
        print(f"  {path} ({size:.1f} MiB)")


def main():
    config_path = os.path.join(_repo_root(), "config.toml")
    args = _settings(_load_config(config_path))

    rank = os.environ.get("ENCODE_RANK")
    world_size = os.environ.get("ENCODE_WORLD_SIZE")
    if rank is not None or world_size is not None:
        if rank is None or world_size is None:
            raise ValueError("ENCODE_RANK and ENCODE_WORLD_SIZE must be set together")
        args.rank = int(rank)
        args.world_size = int(world_size)
        step_encode(args)
        return

    encode_all(args, config_path)


if __name__ == "__main__":
    main()
