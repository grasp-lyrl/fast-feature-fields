import argparse
from dataclasses import dataclass
from timeit import default_timer as timer

import torch
import yaml

from f3 import init_event_model
from f3.event_FF import EventPatchFF
from f3.utils import setup_torch


@dataclass
class BenchResult:
    subsampling: int
    events: int
    hash_ms: float
    full_ms: float


def build_model(config_path: str, device: torch.device) -> torch.nn.Module:
    with open(config_path, "r", encoding="utf-8") as f:
        conf = yaml.safe_load(f)

    if conf.get("model") != "EventPatchFF":
        raise ValueError("This benchmark only supports EventPatchFF configs.")

    # Match test_speed.py behavior: initialize EventFF through init_event_model.
    model = init_event_model(config_path, return_feat=True).to(device)
    if not isinstance(model, EventPatchFF):
        raise TypeError("Loaded model is not EventPatchFF.")

    model.eval()
    return model


def make_events(num_events: int, device: torch.device) -> tuple[torch.Tensor, torch.Tensor]:
    # Mirror test_speed.py synthetic event generation.
    events = torch.rand(num_events, 4, device=device) * 0.9
    events[:, 3] = torch.randint(0, 2, (num_events,), device=device).float()
    counts = torch.tensor([num_events], device=device)
    return events, counts


def benchmark(fn, warmup: int, runs: int) -> float:
    with torch.amp.autocast(device_type="cuda", dtype=torch.float16, enabled=True):
        with torch.no_grad():
            for _ in range(warmup):
                fn()
            torch.cuda.synchronize()
            t0 = timer()
            for _ in range(runs):
                fn()
            torch.cuda.synchronize()
            t1 = timer()
    return (t1 - t0) * 1000.0 / runs


def print_results(results: list[BenchResult]) -> None:
    base_hash = results[0].hash_ms
    base_full = results[0].full_ms

    print("\n=== Subsampling Speed Benchmark (lower is better) ===")
    print(
        f"{'Subsample':>10} {'Events':>10} {'Hash ms':>12} {'Hash speedup':>14} "
        f"{'Full ms':>12} {'Full speedup':>14}"
    )
    print("-" * 78)
    for r in results:
        hash_speedup = base_hash / r.hash_ms
        full_speedup = base_full / r.full_ms
        print(
            f"{r.subsampling:>10d} {r.events:>10d} {r.hash_ms:>12.3f} {hash_speedup:>14.3f} "
            f"{r.full_ms:>12.3f} {full_speedup:>14.3f}"
        )


def main() -> None:
    parser = argparse.ArgumentParser(description="Benchmark full F3 model and internal hash encoder vs subsampling.")
    parser.add_argument(
        "--config",
        type=str,
        default="confs/ff/modeloptions/1280x720x20_patchff_ds1_small.yml",
        help="Path to model config file.",
    )
    parser.add_argument("--base-events", type=int, default=1_000_000, help="Number of events for subsampling=1.")
    parser.add_argument("--rates", type=int, nargs="+", default=[1, 2, 4, 8, 10], help="Subsampling rates.")
    parser.add_argument("--warmup", type=int, default=10, help="Warmup runs per benchmark.")
    parser.add_argument("--runs", type=int, default=30, help="Timed runs per benchmark.")
    args = parser.parse_args()

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for this benchmark script.")

    setup_torch(cudnn_benchmark=True)
    device = torch.device("cuda")
    model = build_model(args.config, device)

    compiled_model = torch.compile(
        model,
        fullgraph=False,
        backend="inductor",
        options={
            "epilogue_fusion": True,
            "max_autotune": True,
        },
    )
    compiled_hash_encoder = torch.compile(
        model.multi_hash_encoder,
        fullgraph=False,
        backend="inductor",
        options={
            "epilogue_fusion": True,
            "max_autotune": True,
        },
    )

    def full_forward(ev: torch.Tensor, cnt: torch.Tensor):
        # Feature path only; output value is intentionally ignored for timing.
        compiled_model(ev, cnt)

    def hash_forward(ev: torch.Tensor):
        compiled_hash_encoder(ev.unsqueeze(0))

    results: list[BenchResult] = []
    print(f"Config: {args.config}")
    print(f"Base events @ subsampling=1: {args.base_events}")
    print(f"Rates: {args.rates}")
    print(f"Warmup runs: {args.warmup}, timed runs: {args.runs}")
    print("Compiled: True")

    for rate in args.rates:
        n_events = max(1, args.base_events // rate)
        events, counts = make_events(n_events, device)

        hash_ms = benchmark(lambda: hash_forward(events), args.warmup, args.runs)
        full_ms = benchmark(lambda: full_forward(events, counts), args.warmup, args.runs)

        results.append(BenchResult(subsampling=rate, events=n_events, hash_ms=hash_ms, full_ms=full_ms))
        print(
            f"Done rate={rate:>2d} events={n_events:>8d} | hash={hash_ms:>8.3f} ms | full={full_ms:>8.3f} ms"
        )

        del events, counts
        torch.cuda.empty_cache()

    print_results(results)


if __name__ == "__main__":
    main()