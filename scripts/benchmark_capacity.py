#!/usr/bin/env python3
"""Measure processing cost of one detect request and write a capacity report.

The report is for sizing a production deployment of ``POST /detect``.
It runs the same ``run_detect`` path the API uses, on this machine, and
records cold start, steady-state latency, CPU time, and resident memory.

Re-run it on the machine you will deploy to. The numbers below are only
true for the host that produced the file.

```bash
uv run python scripts/benchmark_capacity.py
uv run python scripts/benchmark_capacity.py --skip-ocr --skip-sweep
uv run python scripts/benchmark_capacity.py --from-json reports/processing_capacity.json
```

Writes ``reports/processing_capacity.pdf`` (print of the HTML), plus the HTML,
Markdown, and JSON next to it.
"""

from __future__ import annotations

import argparse
import html
import json
import math
import os
import platform
import resource
import shutil
import subprocess
import sys
import tempfile
import threading
import time
from datetime import datetime
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png"}
DEFAULT_DIRS = [ROOT / "SAMPLE IMAGES", ROOT / "totest"]
DEFAULT_PDF = ROOT / "reports" / "processing_capacity.pdf"
DEFAULT_JSON = ROOT / "reports" / "processing_capacity.json"
CHROME_CANDIDATES = (
    Path("/Applications/Google Chrome.app/Contents/MacOS/Google Chrome"),
    Path("/Applications/Chromium.app/Contents/MacOS/Chromium"),
    Path("/usr/bin/google-chrome"),
    Path("/usr/bin/google-chrome-stable"),
    Path("/usr/bin/chromium"),
    Path("/usr/bin/chromium-browser"),
)
BLAS_VARS = ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS")
PACKAGE_NAMES = (
    "torch",
    "torchvision",
    "transformers",
    "ultralytics",
    "opencv-python",
    "opencv-python-headless",
    "numpy",
    "pillow",
    "fastapi",
    "uvicorn",
    "paddlepaddle",
    "paddleocr",
)


def main() -> None:
    args = parse_args()
    if args.worker:
        run_worker(args)
    elif args.from_json:
        payload = json.loads(args.from_json.read_text(encoding="utf-8"))
        publish(payload, args.output, args.json)
    else:
        run_parent(args)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--dir", action="append", dest="dirs", type=Path, help="Photo folder. Repeatable. Default: SAMPLE IMAGES and totest.")
    parser.add_argument("--limit", type=int, default=0, help="Cap the number of photos. 0 means all.")
    parser.add_argument("--sweep-threads", default="4", help="Comma-separated extra STOCK_NUM_THREADS values to compare. Empty skips the sweep.")
    parser.add_argument("--skip-sweep", action="store_true", help="Only measure STOCK_NUM_THREADS=1.")
    parser.add_argument("--skip-ocr", action="store_true", help="Do not measure STOCK_OCR=1.")
    parser.add_argument("--ocr-images", type=int, default=4, help="How many photos to run with OCR on.")
    parser.add_argument("--output", type=Path, default=DEFAULT_PDF, help="PDF report path. HTML and Markdown are written beside it.")
    parser.add_argument("--json", type=Path, default=DEFAULT_JSON)
    parser.add_argument("--from-json", type=Path, help="Render the PDF from a previous measurements file, without re-running.")
    parser.add_argument("--worker", action="store_true", help=argparse.SUPPRESS)
    parser.add_argument("--out", type=Path, help=argparse.SUPPRESS)
    parser.add_argument("--threads", type=int, default=1, help=argparse.SUPPRESS)
    parser.add_argument("--ocr", type=int, choices=(0, 1), default=0, help=argparse.SUPPRESS)
    parser.add_argument("--image-list", type=Path, help=argparse.SUPPRESS)
    return parser.parse_args()


def run_parent(args: argparse.Namespace) -> None:
    images = discover_images(args.dirs or DEFAULT_DIRS, args.limit)
    if not images:
        sys.exit("no JPEG or PNG photos found")

    started = datetime.now().astimezone().isoformat(timespec="seconds")
    print(f"capacity benchmark: {len(images)} photos", file=sys.stderr)
    payload: dict = {
        "generated_at": started,
        "git": git_info(),
        "host": host_info(),
        "disk": disk_info(),
        "photos": [str(path.relative_to(ROOT)) if path.is_relative_to(ROOT) else str(path) for path in images],
        "runs": [],
    }

    payload["runs"].append(spawn_worker(images, threads=1, ocr=0, label="baseline"))

    sweep = [] if args.skip_sweep else parse_thread_list(args.sweep_threads)
    for threads in sweep:
        if threads == 1:
            continue
        payload["runs"].append(spawn_worker(images, threads=threads, ocr=0, label=f"threads-{threads}"))

    if not args.skip_ocr:
        ocr_images = pick_ocr_images(images, args.ocr_images)
        if ocr_images:
            try:
                payload["runs"].append(spawn_worker(ocr_images, threads=1, ocr=1, label="ocr"))
            except subprocess.CalledProcessError as exc:
                payload["ocr_error"] = (exc.stderr or exc.stdout or str(exc))[-4000:]
                print("OCR measurement failed; the report records the error", file=sys.stderr)

    payload["finished_at"] = datetime.now().astimezone().isoformat(timespec="seconds")
    payload["summary"] = summarize(payload)

    publish(payload, args.output, args.json)


def parse_thread_list(raw: str) -> list[int]:
    values = []
    for part in raw.split(","):
        part = part.strip()
        if not part:
            continue
        count = int(part)
        if count < 1:
            raise SystemExit(f"--sweep-threads values must be positive, got {count}")
        values.append(count)
    return values


def discover_images(dirs: list[Path], limit: int) -> list[Path]:
    found: list[Path] = []
    for folder in dirs:
        if not folder.is_dir():
            print(f"skipping missing folder {folder}", file=sys.stderr)
            continue
        for path in sorted(folder.rglob("*")):
            if path.is_file() and path.suffix.lower() in IMAGE_SUFFIXES and not path.name.startswith("."):
                found.append(path.resolve())
    if limit > 0:
        return found[:limit]
    return found


def pick_ocr_images(images: list[Path], count: int) -> list[Path]:
    preferred = [path for path in images if "SAMPLE IMAGES" in path.parts]
    chosen = preferred or images
    return chosen[:count]


def spawn_worker(images: list[Path], *, threads: int, ocr: int, label: str) -> dict:
    print(f"measuring {label}: threads={threads} ocr={ocr} photos={len(images)}", file=sys.stderr)
    fd, list_name = tempfile.mkstemp(prefix="capacity-images-", suffix=".txt")
    out_fd, out_name = tempfile.mkstemp(prefix="capacity-run-", suffix=".json")
    os.close(out_fd)
    list_path = Path(list_name)
    out_path = Path(out_name)
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as handle:
            handle.write("\n".join(str(path) for path in images))
            handle.write("\n")
        env = os.environ.copy()
        env["STOCK_NUM_THREADS"] = str(threads)
        env["STOCK_OCR"] = "1" if ocr else "0"
        env["LOG_LEVEL"] = "ERROR"
        env["PYTHONUNBUFFERED"] = "1"
        for key in BLAS_VARS:
            env[key] = str(threads)
        cmd = [
            sys.executable,
            str(Path(__file__).resolve()),
            "--worker",
            "--out",
            str(out_path),
            "--threads",
            str(threads),
            "--ocr",
            str(ocr),
            "--image-list",
            str(list_path),
        ]
        proc = subprocess.Popen(
            cmd,
            cwd=ROOT,
            env=env,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
        )
        log: list[str] = []
        assert proc.stdout is not None
        for line in proc.stdout:
            log.append(line)
            print(line, end="", file=sys.stderr)
        code = proc.wait()
        if code != 0:
            tail = "".join(log[-40:])
            raise subprocess.CalledProcessError(code, cmd, output=tail, stderr=tail)
        data = json.loads(out_path.read_text(encoding="utf-8"))
        data["label"] = label
        return data
    finally:
        list_path.unlink(missing_ok=True)
        out_path.unlink(missing_ok=True)


def run_worker(args: argparse.Namespace) -> None:
    os.environ["STOCK_NUM_THREADS"] = str(args.threads)
    os.environ["STOCK_OCR"] = "1" if args.ocr else "0"
    for key in BLAS_VARS:
        os.environ[key] = str(args.threads)
    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

    images = [Path(line) for line in args.image_list.read_text(encoding="utf-8").splitlines() if line.strip()]
    probe = RssProbe()
    probe.start()
    started = time.perf_counter()

    t0 = time.perf_counter()
    import torch

    from stock_material_detection.cpu_threads import resolve_num_threads
    from stock_material_detection.detector import (
        FALLBACK_PROMPTS,
        IMAGE_SIZE,
        PROMPTS,
        Detector,
    )
    from stock_material_detection.gallery import MODEL_ID, Gallery, WordmarkReader
    from stock_material_detection.pipeline import get_detector, get_gallery, get_ocr, run_detect
    import stock_material_detection.pipeline as pipeline

    import_s = time.perf_counter() - t0

    acc = StageAccum()
    install_hooks(pipeline, Detector, Gallery, WordmarkReader, acc)

    t0 = time.perf_counter()
    detector = get_detector()
    detector_s = time.perf_counter() - t0

    t0 = time.perf_counter()
    gallery = get_gallery()
    gallery_index_s = time.perf_counter() - t0

    t0 = time.perf_counter()
    embedder = gallery.embedder
    embedder_s = time.perf_counter() - t0

    ocr_s = None
    if args.ocr:
        t0 = time.perf_counter()
        get_ocr()
        ocr_s = time.perf_counter() - t0

    rss_after_load = probe.sample()
    ready_s = time.perf_counter() - started
    threads_after_load = torch_thread_state()

    warmup = time_one(images[0], run_detect, acc, probe)
    threads_during_requests = torch_thread_state()
    timed = []
    for index, path in enumerate(images, start=1):
        row = time_one(path, run_detect, acc, probe)
        timed.append(row)
        print(
            f"  [{index}/{len(images)}] {row['name']} {row['wall_s']:.2f}s "
            f"detect={row['detect_s']:.2f}s embed={row['embed_s']:.2f}s boxes={row['detections']}",
            file=sys.stderr,
        )

    result = {
        "threads_requested": args.threads,
        "threads_resolved": resolve_num_threads(),
        "torch_threads_after_load": threads_after_load,
        "torch_threads_during_requests": threads_during_requests,
        "ultralytics_num_threads": ultralytics_num_threads(),
        "ocr": bool(args.ocr),
        "import_s": import_s,
        "load_detector_s": detector_s,
        "load_gallery_index_s": gallery_index_s,
        "load_embedder_s": embedder_s,
        "load_ocr_s": ocr_s,
        "ready_s": ready_s,
        "rss_after_load": rss_after_load,
        "backend": detector.backend,
        "visual_categories": list(detector.visual_categories),
        "device": "cpu",
        "mps_available": bool(torch.backends.mps.is_available()),
        "cuda_available": bool(torch.cuda.is_available()),
        "models": model_facts(detector, embedder),
        "gallery": {
            "references": len(gallery.entries),
            "skus": len(set(gallery.skus.tolist())),
            "embed_dim": int(gallery.vectors.shape[1]) if gallery.vectors.ndim == 2 else 0,
        },
        "config": {
            "detector_image_size": IMAGE_SIZE,
            "text_prompts": len(PROMPTS),
            "fallback_prompts": len(FALLBACK_PROMPTS),
            "siglip_id": MODEL_ID,
            "max_upload_bytes": 25 * 1024 * 1024,
        },
        "packages": package_versions(),
        "warmup": warmup,
        "images": timed,
        "rss_after_requests": probe.sample(),
        "rss_peak_sampled": probe.peak,
        "maxrss": maxrss_bytes(),
        "elapsed_s": time.perf_counter() - started,
    }
    probe.stop()
    args.out.write_text(json.dumps(result), encoding="utf-8")


class StageAccum:
    def __init__(self) -> None:
        self.reset()

    def reset(self) -> None:
        self.orient_s = 0.0
        self.detect_s = 0.0
        self.embed_s = 0.0
        self.embed_calls = 0
        self.embed_views = 0
        self.match_s = 0.0
        self.ocr_s = 0.0
        self.ocr_calls = 0


def install_hooks(pipeline, detector_cls, gallery_cls, ocr_cls, acc: StageAccum) -> None:
    orig_load = pipeline.load_oriented
    orig_bgr = pipeline.to_bgr
    orig_detect = detector_cls.detect
    orig_embed = gallery_cls.embed
    orig_match = gallery_cls.match
    orig_check = ocr_cls.check

    def load_oriented(path):
        t0 = time.perf_counter()
        try:
            return orig_load(path)
        finally:
            acc.orient_s += time.perf_counter() - t0

    def to_bgr(image):
        t0 = time.perf_counter()
        try:
            return orig_bgr(image)
        finally:
            acc.orient_s += time.perf_counter() - t0

    def detect(self, image, **kwargs):
        t0 = time.perf_counter()
        try:
            return orig_detect(self, image, **kwargs)
        finally:
            acc.detect_s += time.perf_counter() - t0

    def embed(self, crop):
        t0 = time.perf_counter()
        try:
            return orig_embed(self, crop)
        finally:
            acc.embed_s += time.perf_counter() - t0
            acc.embed_calls += 1
            acc.embed_views += 4

    def match(self, vectors, hist=None):
        t0 = time.perf_counter()
        try:
            return orig_match(self, vectors, hist)
        finally:
            acc.match_s += time.perf_counter() - t0

    def check(self, crop, wordmark):
        t0 = time.perf_counter()
        try:
            return orig_check(self, crop, wordmark)
        finally:
            acc.ocr_s += time.perf_counter() - t0
            acc.ocr_calls += 1

    pipeline.load_oriented = load_oriented
    pipeline.to_bgr = to_bgr
    detector_cls.detect = detect
    gallery_cls.embed = embed
    gallery_cls.match = match
    ocr_cls.check = check


def time_one(path: Path, run_detect, acc: StageAccum, probe: "RssProbe") -> dict:
    acc.reset()
    cpu0 = process_cpu_time()
    wall0 = time.perf_counter()
    error = None
    detections = counts = 0
    try:
        result = run_detect(path)
        detections = len(result.detections)
        counts = len(result.counts)
        if result.failReasons:
            error = ",".join(result.failReasons)
    except Exception as exc:
        error = f"{type(exc).__name__}: {exc}"
    wall_s = time.perf_counter() - wall0
    cpu_s = process_cpu_time() - cpu0
    width = height = None
    try:
        from PIL import Image, ImageOps

        with Image.open(path) as im:
            width, height = ImageOps.exif_transpose(im).size
    except OSError:
        pass
    accounted = acc.orient_s + acc.detect_s + acc.embed_s + acc.match_s + acc.ocr_s
    return {
        "name": path.name,
        "path": str(path),
        "bytes": path.stat().st_size if path.is_file() else None,
        "width": width,
        "height": height,
        "wall_s": wall_s,
        "cpu_s": cpu_s,
        "orient_s": acc.orient_s,
        "detect_s": acc.detect_s,
        "embed_s": acc.embed_s,
        "match_s": acc.match_s,
        "ocr_s": acc.ocr_s,
        "other_s": wall_s - accounted,
        "embed_calls": acc.embed_calls,
        "embed_views": acc.embed_views,
        "ocr_calls": acc.ocr_calls,
        "detections": detections,
        "counts": counts,
        "rss": probe.sample(),
        "error": error,
    }


def model_facts(detector, embedder: object) -> dict:
    yolo = getattr(detector.model, "model", None)
    siglip = getattr(embedder, "model", None)
    facts = {
        "yolo_bytes": module_bytes(yolo),
        "yolo_params": module_params(yolo),
        "siglip_bytes": module_bytes(siglip),
        "siglip_params": module_params(siglip),
        "siglip_groups": param_groups(siglip),
        "siglip_dtype": dtype_name(siglip),
    }
    return facts


def module_params(module) -> int:
    if module is None:
        return 0
    return sum(p.numel() for p in module.parameters())


def module_bytes(module) -> int:
    if module is None:
        return 0
    total = 0
    for tensor in list(module.parameters()) + list(module.buffers()):
        total += tensor.numel() * tensor.element_size()
    return total


def param_groups(module) -> dict[str, int]:
    if module is None:
        return {}
    groups: dict[str, int] = {}
    for name, param in module.named_parameters():
        top = name.split(".", 1)[0]
        groups[top] = groups.get(top, 0) + param.numel()
    return dict(sorted(groups.items(), key=lambda item: -item[1]))


def dtype_name(module) -> str | None:
    if module is None:
        return None
    for param in module.parameters():
        return str(param.dtype).replace("torch.", "")
    return None


def torch_thread_state() -> dict:
    import torch

    state = {"intra_op": torch.get_num_threads()}
    try:
        state["inter_op"] = torch.get_num_interop_threads()
    except RuntimeError:
        state["inter_op"] = None
    try:
        import cv2

        state["opencv"] = cv2.getNumThreads()
    except Exception:
        state["opencv"] = None
    return state


def ultralytics_num_threads() -> int | None:
    try:
        from ultralytics.utils import NUM_THREADS
    except Exception:
        return None
    return int(NUM_THREADS)


def package_versions() -> dict[str, str]:
    found = {}
    for name in PACKAGE_NAMES:
        try:
            found[name] = version(name)
        except PackageNotFoundError:
            continue
    return found


class RssProbe:
    def __init__(self) -> None:
        self.peak = 0
        self._stop = threading.Event()
        self._thread = threading.Thread(target=self._loop, name="rss-probe", daemon=True)

    def start(self) -> None:
        self.sample()
        self._thread.start()

    def stop(self) -> None:
        self._stop.set()
        self._thread.join(timeout=1)
        self.sample()

    def sample(self) -> int:
        value = current_rss()
        if value > self.peak:
            self.peak = value
        return value

    def _loop(self) -> None:
        while not self._stop.wait(0.1):
            try:
                self.sample()
            except OSError:
                return


def current_rss() -> int:
    if sys.platform.startswith("linux"):
        for line in Path("/proc/self/status").read_text(encoding="utf-8").splitlines():
            if line.startswith("VmRSS:"):
                return int(line.split()[1]) * 1024
        raise OSError("VmRSS missing from /proc/self/status")
    raw = subprocess.check_output(["ps", "-o", "rss=", "-p", str(os.getpid())], text=True)
    return int(raw.strip().splitlines()[-1]) * 1024


def maxrss_bytes() -> int:
    rss = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    if sys.platform == "darwin":
        return int(rss)
    return int(rss) * 1024


def process_cpu_time() -> float:
    usage = resource.getrusage(resource.RUSAGE_SELF)
    return usage.ru_utime + usage.ru_stime


def host_info() -> dict:
    info = {
        "hostname": platform.node(),
        "platform": platform.platform(),
        "machine": platform.machine(),
        "python": platform.python_version(),
        "logical_cpus": os.cpu_count(),
    }
    if sys.platform == "darwin":
        info["cpu"] = sysctl("machdep.cpu.brand_string")
        mem = sysctl("hw.memsize")
        info["ram_bytes"] = int(mem) if mem else None
        perf = sysctl("hw.perflevel0.physicalcpu")
        eff = sysctl("hw.perflevel1.physicalcpu")
        if perf or eff:
            info["performance_cores"] = int(perf) if perf else None
            info["efficiency_cores"] = int(eff) if eff else None
    elif Path("/proc/meminfo").is_file():
        info.update(linux_host())
    return info


def linux_host() -> dict:
    info: dict = {}
    for line in Path("/proc/meminfo").read_text(encoding="utf-8").splitlines():
        if line.startswith("MemTotal:"):
            info["ram_bytes"] = int(line.split()[1]) * 1024
            break
    cpuinfo = Path("/proc/cpuinfo").read_text(encoding="utf-8", errors="replace")
    for line in cpuinfo.splitlines():
        if line.startswith("model name"):
            info["cpu"] = line.split(":", 1)[1].strip()
            break
    return info


def sysctl(name: str) -> str:
    try:
        return subprocess.check_output(["sysctl", "-n", name], text=True).strip()
    except (OSError, subprocess.CalledProcessError):
        return ""


def git_info() -> dict:
    def run(*args: str) -> str:
        return subprocess.check_output(["git", *args], cwd=ROOT, text=True).strip()

    try:
        return {
            "sha": run("rev-parse", "HEAD"),
            "short": run("rev-parse", "--short", "HEAD"),
            "branch": run("rev-parse", "--abbrev-ref", "HEAD"),
            "dirty": bool(run("status", "--porcelain")),
        }
    except (OSError, subprocess.CalledProcessError):
        return {}


def disk_info() -> dict:
    weights = ROOT / "weights"
    visual = ROOT / "visual_prompts"
    gallery = ROOT / "gallery"
    hf_hub = Path(os.environ.get("HF_HOME", Path.home() / ".cache" / "huggingface")) / "hub"
    items = []
    if weights.is_dir():
        for path in sorted(weights.iterdir()):
            if path.is_file():
                items.append(artifact(path.name, path, "weights"))
    if visual.is_dir():
        for path in sorted(visual.rglob("*")):
            if path.is_file() and not path.name.startswith("."):
                items.append(artifact(str(path.relative_to(ROOT)), path, "visual prompts"))
    cache = gallery / ".embeddings.npz"
    if cache.is_file():
        items.append(artifact("gallery/.embeddings.npz", cache, "gallery vectors"))
    items.append(
        {
            "name": "gallery reference images",
            "role": "rebuilt into the vector cache when a file changes",
            "bytes": tree_bytes(gallery, skip_names={".embeddings.npz"}),
        }
    )
    siglip = hf_hub / "models--google--siglip2-base-patch16-224"
    items.append(
        {
            "name": "google/siglip2-base-patch16-224",
            "role": "Hugging Face cache, loaded at startup",
            "bytes": unique_bytes(siglip),
            "present": siglip.is_dir(),
        }
    )
    paddlex = Path.home() / ".paddlex"
    items.append(
        {
            "name": "PaddleOCR official models",
            "role": "downloaded on first STOCK_OCR=1 run",
            "bytes": unique_bytes(paddlex),
            "present": paddlex.is_dir(),
        }
    )
    packages = package_disk()
    return {"artifacts": items, "python_packages": packages, "python_packages_bytes": sum(row["bytes"] for row in packages)}


def artifact(name: str, path: Path, role: str) -> dict:
    return {"name": name, "role": role, "bytes": path.stat().st_size, "present": True}


def tree_bytes(root: Path, *, skip_names: set[str] | None = None) -> int:
    skip = skip_names or set()
    total = 0
    if not root.is_dir():
        return 0
    for path in root.rglob("*"):
        if path.is_file() and path.name not in skip and not path.name.startswith("."):
            total += path.stat().st_size
    return total


def unique_bytes(root: Path) -> int:
    if not root.exists():
        return 0
    seen: set[Path] = set()
    total = 0
    for path in root.rglob("*"):
        if not path.is_file() and not path.is_symlink():
            continue
        try:
            resolved = path.resolve()
            if not resolved.is_file() or resolved in seen:
                continue
            seen.add(resolved)
            total += resolved.stat().st_size
        except OSError:
            continue
    return total


def package_disk() -> list[dict]:
    site = next((Path(entry) for entry in sys.path if Path(entry).name == "site-packages"), None)
    if site is None:
        return []
    rows = []
    for name in (
        "torch",
        "torchvision",
        "transformers",
        "ultralytics",
        "cv2",
        "numpy",
        "PIL",
        "paddle",
        "paddleocr",
        "fastapi",
        "uvicorn",
    ):
        path = site / name
        if path.exists():
            rows.append({"name": name, "bytes": unique_bytes(path)})
    return rows


def summarize(payload: dict) -> dict:
    baseline = run_by_label(payload, "baseline")
    rows = ok_rows(baseline)
    walls = [row["wall_s"] for row in rows]
    summary = {
        "photos": len(rows),
        "failed": sum(1 for row in baseline.get("images", []) if row.get("error")),
        "ready_s": baseline.get("ready_s"),
        "first_request_s": (baseline.get("warmup") or {}).get("wall_s"),
        "wall_mean_s": mean(walls),
        "wall_p50_s": percentile(walls, 0.50),
        "wall_p90_s": percentile(walls, 0.90),
        "wall_p95_s": percentile(walls, 0.95),
        "wall_max_s": max(walls) if walls else None,
        "per_minute_at_mean": rate(mean(walls)),
        "per_minute_at_p95": rate(percentile(walls, 0.95)),
        "stages": stage_means(rows),
        "effective_cores": mean([row["cpu_s"] / row["wall_s"] for row in rows if row["wall_s"] > 0]),
        "mean_detections": mean([row["detections"] for row in rows]),
        "mean_embed_calls": mean([row["embed_calls"] for row in rows]),
        "rss_after_load": baseline.get("rss_after_load"),
        "maxrss": baseline.get("maxrss"),
        "thread_options": [],
        "correlations": {
            "wall_vs_detections": correlate([row["detections"] for row in rows], walls),
            "wall_vs_megapixels": correlate([megapixels(row) for row in rows], walls),
        },
    }
    for run in payload["runs"]:
        if run.get("ocr"):
            continue
        run_rows = ok_rows(run)
        run_walls = [row["wall_s"] for row in run_rows]
        during = run.get("torch_threads_during_requests") or {}
        summary["thread_options"].append(
            {
                "threads": run["threads_resolved"],
                "torch_intra_op": during.get("intra_op"),
                "torch_inter_op": during.get("inter_op"),
                "opencv_threads": during.get("opencv"),
                "ultralytics_num_threads": run.get("ultralytics_num_threads"),
                "photos": len(run_rows),
                "wall_mean_s": mean(run_walls),
                "wall_p95_s": percentile(run_walls, 0.95),
                "per_minute": rate(mean(run_walls)),
                "effective_cores": mean(
                    [row["cpu_s"] / row["wall_s"] for row in run_rows if row["wall_s"] > 0]
                ),
                "maxrss": run.get("maxrss"),
                "ready_s": run.get("ready_s"),
            }
        )
    ocr = run_by_label(payload, "ocr")
    if ocr:
        summary["ocr"] = ocr_delta(baseline, ocr)
    summary["torch_threads_after_load"] = (baseline or {}).get("torch_threads_after_load")
    summary["torch_threads_during_requests"] = (baseline or {}).get("torch_threads_during_requests")
    summary["ultralytics_num_threads"] = (baseline or {}).get("ultralytics_num_threads")
    summary["sizing"] = sizing(payload, summary)
    return summary


def ocr_delta(baseline: dict, ocr: dict) -> dict:
    base = {row["name"]: row for row in ok_rows(baseline)}
    deltas = []
    for row in ok_rows(ocr):
        other = base.get(row["name"])
        if other is None:
            continue
        deltas.append(
            {
                "name": row["name"],
                "wall_s": row["wall_s"],
                "base_wall_s": other["wall_s"],
                "delta_s": row["wall_s"] - other["wall_s"],
                "ocr_s": row["ocr_s"],
                "ocr_calls": row["ocr_calls"],
            }
        )
    delta_vals = [row["delta_s"] for row in deltas]
    return {
        "photos": len(deltas),
        "mean_delta_s": mean(delta_vals),
        "mean_ocr_s": mean([row["ocr_s"] for row in deltas]),
        "ready_s": ocr.get("ready_s"),
        "load_ocr_s": ocr.get("load_ocr_s"),
        "rss_after_load": ocr.get("rss_after_load"),
        "maxrss": ocr.get("maxrss"),
        "images": deltas,
    }


def sizing(payload: dict, summary: dict) -> dict:
    host_ram = (payload.get("host") or {}).get("ram_bytes")
    peak = summary.get("maxrss") or 0
    gib = recommend_gib(peak)
    cores = summary.get("effective_cores") or 0
    vcpu = max(1, math.ceil(cores - 1e-9)) if cores else 1
    p95 = summary.get("wall_p95_s") or 0
    return {
        "peak_rss_bytes": peak,
        "recommend_ram_gib": gib,
        "recommend_vcpu": vcpu,
        "host_ram_bytes": host_ram,
        "memory_pressure": bool(host_ram and peak > 0.75 * host_ram),
        "startup_probe_s": max(60, math.ceil((summary.get("ready_s") or 0) * 2)),
        "request_timeout_s": max(15, math.ceil(p95 * 4)) if p95 else 30,
        "examples": [
            {"photos_per_minute": rate, "replicas_at_p95": replicas_for(rate, p95)}
            for rate in (10, 30, 60, 120)
        ],
    }


def recommend_gib(peak: int) -> float:
    if peak <= 0:
        return 0
    need = peak * 1.5
    step = 0.5 * 1024**3
    return math.ceil(need / step) * 0.5


def replicas_for(photos_per_minute: float, p95_s: float) -> int:
    if p95_s <= 0:
        return 0
    return max(1, math.ceil((photos_per_minute / 60) * p95_s))


def stage_means(rows: list[dict]) -> dict[str, float | None]:
    keys = ("orient_s", "detect_s", "embed_s", "match_s", "ocr_s", "other_s")
    return {key: mean([row[key] for row in rows]) for key in keys}


def ok_rows(run: dict | None) -> list[dict]:
    if not run:
        return []
    return [row for row in run.get("images", []) if not row.get("error")]


def run_by_label(payload: dict, label: str) -> dict | None:
    for run in payload["runs"]:
        if run.get("label") == label:
            return run
    return None


def megapixels(row: dict) -> float:
    if not row.get("width") or not row.get("height"):
        return 0.0
    return row["width"] * row["height"] / 1_000_000


def mean(values: list[float]) -> float | None:
    if not values:
        return None
    return sum(values) / len(values)


def rate(seconds: float | None) -> float | None:
    if not seconds or seconds <= 0:
        return None
    return 60 / seconds


def percentile(values: list[float], p: float) -> float | None:
    if not values:
        return None
    ordered = sorted(values)
    if len(ordered) == 1:
        return ordered[0]
    rank = (len(ordered) - 1) * p
    lo = int(math.floor(rank))
    hi = min(lo + 1, len(ordered) - 1)
    return ordered[lo] + (ordered[hi] - ordered[lo]) * (rank - lo)


def correlate(xs: list[float], ys: list[float]) -> float | None:
    pairs = [(x, y) for x, y in zip(xs, ys)]
    if len(pairs) < 3:
        return None
    mx = sum(x for x, _ in pairs) / len(pairs)
    my = sum(y for _, y in pairs) / len(pairs)
    num = sum((x - mx) * (y - my) for x, y in pairs)
    dx = math.sqrt(sum((x - mx) ** 2 for x, _ in pairs))
    dy = math.sqrt(sum((y - my) ** 2 for _, y in pairs))
    if dx == 0 or dy == 0:
        return None
    return num / (dx * dy)


def render_markdown(payload: dict) -> str:
    summary = payload["summary"]
    baseline = run_by_label(payload, "baseline")
    host = payload["host"]
    lines: list[str] = []
    add = lines.append

    add("# Processing capacity — stock material detection")
    add("")
    add(f"Generated {payload['generated_at']} by `scripts/benchmark_capacity.py`.")
    git = payload.get("git") or {}
    if git:
        dirty = ", uncommitted changes" if git.get("dirty") else ""
        add(f"Code `{git.get('short', '')}` on `{git.get('branch', '')}`{dirty}.")
    add("")
    add("These numbers are measurements from the host below, not a portable budget.")
    add("Run the same script on the production machine and hand that file over.")
    add("")
    add("```bash")
    add("uv run python scripts/benchmark_capacity.py")
    add("```")
    add("")
    add("## What to provision")
    add("")
    sizing_block(add, payload, summary)
    add("")
    add("## Host")
    add("")
    add("| | |")
    add("|---|---|")
    add(f"| Machine | {host.get('hostname', '')} |")
    add(f"| OS | {host.get('platform', '')} |")
    cpu = host.get("cpu") or host.get("machine")
    cores = host.get("logical_cpus")
    perf = host.get("performance_cores")
    eff = host.get("efficiency_cores")
    core_note = f"{cores} logical"
    if perf or eff:
        core_note += f", {perf or 0} performance + {eff or 0} efficiency"
    add(f"| CPU | {cpu} ({core_note}) |")
    add(f"| RAM | {fmt_bytes(host.get('ram_bytes'))} |")
    add(f"| Python | {host.get('python')} |")
    if baseline:
        pkgs = baseline.get("packages") or {}
        for name in ("torch", "transformers", "ultralytics", "paddlepaddle", "paddleocr"):
            if name in pkgs:
                add(f"| {name} | {pkgs[name]} |")
    add("")
    add("The detector calls Ultralytics with `device=\"cpu\"`. MPS or CUDA being present does not move this pipeline off CPU.")
    if baseline:
        add(
            f"This run saw MPS {'available' if baseline.get('mps_available') else 'unavailable'}"
            f" and CUDA {'available' if baseline.get('cuda_available') else 'unavailable'}."
        )
    add("")
    add("## What one request costs")
    add("")
    add("One photo is decoded, EXIF-rotated, and run through three models in series:")
    add("")
    add("1. YOLOE (`yoloe-11l-seg`) boxes the photo. The long side is resized to 960 px, so a larger upload does not make the detector slower.")
    add("2. Each kept box is cropped. SigLIP 2 embeds that crop in four quarter-turns and cosine-ranks it against the gallery.")
    add("3. With `STOCK_OCR=1`, PaddleOCR reads the wordmark on pack, lighter, and earphones crops, again in up to four turns.")
    add("")
    if baseline:
        cfg = baseline["config"]
        gal = baseline["gallery"]
        add(
            f"This gallery has {gal['skus']} SKUs and {gal['references']} reference images, "
            f"embed dim {gal['embed_dim']}. The detector has {cfg['text_prompts']} text prompts and "
            f"{cfg['fallback_prompts']} fallback prompts"
            f"{', plus visual prompts for ' + ', '.join(baseline['visual_categories']) if baseline.get('visual_categories') else ''}."
        )
        add(f"Backend loaded: `{baseline['backend']}`.")
        add("")
        models = baseline.get("models") or {}
        add("| Weights resident after load | Parameters | Bytes |")
        add("|---|---:|---:|")
        add(f"| YOLOE | {fmt_count(models.get('yolo_params'))} | {fmt_bytes(models.get('yolo_bytes'))} |")
        add(f"| SigLIP 2 (`{cfg['siglip_id']}`, {models.get('siglip_dtype')}) | {fmt_count(models.get('siglip_params'))} | {fmt_bytes(models.get('siglip_bytes'))} |")
        groups = models.get("siglip_groups") or {}
        if groups:
            add("")
            add("SigLIP parameter groups: " + ", ".join(f"{name} {fmt_count(count)}" for name, count in groups.items()) + ".")
            add("Requests only call `get_image_features`. The text tower is still loaded with the checkpoint.")
        add("")
    add(
        f"Steady state over {summary['photos']} photos, `STOCK_NUM_THREADS=1`, OCR off. "
        "The first request after load is excluded from these percentiles and reported on its own."
    )
    add("")
    add("| | |")
    add("|---|---|")
    add(f"| Models ready (import + weights, before any photo) | {fmt_s(summary.get('ready_s'))} |")
    add(f"| First photo after load | {fmt_s(summary.get('first_request_s'))} |")
    add(f"| Mean | {fmt_s(summary.get('wall_mean_s'))} |")
    add(f"| p50 | {fmt_s(summary.get('wall_p50_s'))} |")
    add(f"| p90 | {fmt_s(summary.get('wall_p90_s'))} |")
    add(f"| p95 | {fmt_s(summary.get('wall_p95_s'))} |")
    add(f"| Max | {fmt_s(summary.get('wall_max_s'))} |")
    add(f"| Photos / minute at the mean | {fmt_rate(summary.get('per_minute_at_mean'))} |")
    add(f"| Photos / minute if every photo costs p95 | {fmt_rate(summary.get('per_minute_at_p95'))} |")
    add(f"| Process CPU time / wall time | {fmt_cores(summary.get('effective_cores'))} |")
    add(f"| Mean boxes per photo | {fmt_num(summary.get('mean_detections'))} |")
    add(f"| Mean SigLIP calls per photo | {fmt_num(summary.get('mean_embed_calls'))} |")
    if summary.get("failed"):
        add(f"| Photos that failed | {summary['failed']} |")
    add("")
    add("Stage means, same photos:")
    add("")
    add("| Stage | Mean | Share of mean request |")
    add("|---|---:|---:|")
    wall_mean = summary.get("wall_mean_s") or 0
    stage_labels = (
        ("orient_s", "EXIF rotate and decode"),
        ("detect_s", "YOLOE detect"),
        ("embed_s", "SigLIP embed (4 turns per crop)"),
        ("match_s", "Gallery cosine match"),
        ("ocr_s", "PaddleOCR"),
        ("other_s", "Crops, stick bands, histograms, counting"),
    )
    for key, label in stage_labels:
        value = (summary.get("stages") or {}).get(key)
        add(f"| {label} | {fmt_s(value)} | {fmt_share(value, wall_mean)} |")
    add("")
    corr = summary.get("correlations") or {}
    add(
        f"Wall time correlates with box count at {fmt_corr(corr.get('wall_vs_detections'))} "
        f"and with megapixels at {fmt_corr(corr.get('wall_vs_megapixels'))}. "
        "A busy photo costs more because each box is embedded separately. The detector input size stays 960."
    )
    add("")
    add("## Threads")
    add("")
    add("Startup sets `STOCK_NUM_THREADS` (default 1) on PyTorch, OpenCV, and Paddle, and fills in `OMP_NUM_THREADS`, `MKL_NUM_THREADS`, and `OPENBLAS_NUM_THREADS` only when those variables are unset. This benchmark set all four to the value in the table before importing anything.")
    add("")
    ultra = summary.get("ultralytics_num_threads")
    add(
        "That pin does not survive the first detect. Ultralytics `select_device(\"cpu\")` calls "
        "`torch.set_num_threads(min(8, cpu_count - 1))` on every CPU predict. "
        f"On this host that is {ultra if ultra is not None else 'n/a'} threads. "
        "Size the CPU limit from the measured CPU-seconds, not from `STOCK_NUM_THREADS`."
    )
    add("")
    after = summary.get("torch_threads_after_load") or {}
    during = summary.get("torch_threads_during_requests") or {}
    add(
        f"With `STOCK_NUM_THREADS=1`, PyTorch intra-op threads were {after.get('intra_op', 'n/a')} after load and "
        f"{during.get('intra_op', 'n/a')} while photos were running. "
        f"Inter-op threads during requests: {during.get('inter_op', 'n/a')}. "
        f"OpenCV threads during requests: {during.get('opencv', 'n/a')}."
    )
    add("")
    options = summary.get("thread_options") or []
    if options:
        add("| STOCK_NUM_THREADS | Torch intra-op during requests | OpenCV | Mean | p95 | Photos/min/replica | CPU time / wall | Peak RSS |")
        add("|---:|---:|---:|---:|---:|---:|---:|---:|")
        for option in options:
            add(
                f"| {option['threads']} | {option.get('torch_intra_op')} | {option.get('opencv_threads')} | "
                f"{fmt_s(option['wall_mean_s'])} | {fmt_s(option['wall_p95_s'])} | {fmt_rate(option['per_minute'])} | "
                f"{fmt_cores(option['effective_cores'])} | {fmt_bytes(option['maxrss'])} |"
            )
        add("")
        sweep_note(add, options)
    add("")
    add("## Memory")
    add("")
    add("| | |")
    add("|---|---|")
    add(f"| RSS after models are loaded, before a photo | {fmt_bytes(summary.get('rss_after_load'))} |")
    add(f"| Peak RSS (OS high-water, OCR off) | {fmt_bytes(summary.get('maxrss'))} |")
    if baseline:
        add(f"| RSS after the timed photos | {fmt_bytes(baseline.get('rss_after_requests'))} |")
    ocr = summary.get("ocr")
    if ocr:
        add(f"| RSS after load with OCR | {fmt_bytes(ocr.get('rss_after_load'))} |")
        add(f"| Peak RSS with OCR | {fmt_bytes(ocr.get('maxrss'))} |")
    add("")
    add("The API reads the whole upload (max 25 MiB) before it takes the inference lock. Queued requests hold their bytes at the same time as the model.")
    add("A reverse proxy should cap in-flight requests to this service. Each extra waiting upload adds up to 25 MiB on top of the peak above.")
    add("")
    if ocr:
        add("## OCR")
    add("")
    if ocr:
        add("`STOCK_OCR` defaults to off. The OCR run used 1 thread on a subset of the same photos.")
        add("")
        add("| | |")
        add("|---|---|")
        add(f"| Photos compared | {ocr.get('photos')} |")
        add(f"| PaddleOCR load | {fmt_s(ocr.get('load_ocr_s'))} |")
        add(f"| Models ready, including OCR | {fmt_s(ocr.get('ready_s'))} |")
        add(f"| Extra wall time vs OCR off, mean | {fmt_s(ocr.get('mean_delta_s'))} |")
        add(f"| Time inside PaddleOCR, mean | {fmt_s(ocr.get('mean_ocr_s'))} |")
        add("")
        if ocr.get("images"):
            add("| Photo | OCR off | OCR on | Added | Inside OCR | OCR calls |")
            add("|---|---:|---:|---:|---:|---:|")
            for row in ocr["images"]:
                add(
                    f"| {row['name']} | {fmt_s(row['base_wall_s'])} | {fmt_s(row['wall_s'])} | "
                    f"{fmt_s(row['delta_s'])} | {fmt_s(row['ocr_s'])} | {row['ocr_calls']} |"
                )
            add("")
    elif payload.get("ocr_error"):
        add("## OCR")
        add("")
        add("The OCR measurement failed. Baseline numbers above are OCR off.")
        add("")
        add("```")
        add(payload["ocr_error"].rstrip())
        add("```")
        add("")
    add("## How the process scales")
    add("")
    add("The FastAPI app holds one lock around the whole pipeline. A second photo waits until the first returns. `GET /health` does not answer until startup has finished loading models (`STOCK_WARMUP=1`).")
    add("")
    add("Run one uvicorn process per container, with `--workers 1`. Another worker is another copy of the weights in RAM and another lock; the locks do not coordinate. To run two photos at once, run two containers and give each one the RAM below.")
    add("")
    add("Replicas below assume a load balancer spreads photos evenly and you want the slow tail (p95) to stay at the measured p95, with no queue on the replica.")
    add("")
    add("| Target photos / minute | Replicas at the measured p95 |")
    add("|---:|---:|")
    for example in (summary.get("sizing") or {}).get("examples") or []:
        add(f"| {example['photos_per_minute']} | {example['replicas_at_p95']} |")
    add("")
    p95 = summary.get("wall_p95_s")
    timeout = (summary.get("sizing") or {}).get("request_timeout_s")
    probe = (summary.get("sizing") or {}).get("startup_probe_s")
    add(f"Set the proxy request timeout to at least {timeout} s ({fmt_s(p95)} p95, with room for a short queue).")
    add(f"Set the startup probe to at least {probe} s. Model load measured {fmt_s(summary.get('ready_s'))} on this host.")
    add("")
    add("There is no GPU path in this build. A GPU box does not speed it up until the `device=\"cpu\"` arguments in the detector and the SigLIP call are changed and this script is run again.")
    add("")
    add("## Disk")
    add("")
    add("Weights download on first use into `weights/`, the Hugging Face cache, and `~/.paddlex` when OCR is on. Bake them into the image or a volume so a new container does not download on the readiness path.")
    add("")
    add("| Artifact | Role | Size |")
    add("|---|---|---:|")
    for row in payload["disk"]["artifacts"]:
        present = "" if row.get("present", True) else " (not on this machine)"
        add(f"| {row['name']}{present} | {row['role']} | {fmt_bytes(row['bytes'])} |")
    add("")
    add(f"Python packages in this environment, the ones the process imports: {fmt_bytes(payload['disk']['python_packages_bytes'])}.")
    add("")
    add("| Package directory | Size |")
    add("|---|---:|")
    for row in payload["disk"]["python_packages"]:
        add(f"| {row['name']} | {fmt_bytes(row['bytes'])} |")
    add("")
    add("`yolov8l-worldv2.pt` is loaded only when YOLOE fails to start. A healthy deploy does not need it in memory. It is still a download on the fallback path.")
    add("")
    add("The gallery vector file is rebuilt when a reference image changes. That rebuild runs SigLIP over every reference once, inside the process that noticed the change. Ship an already-built `gallery/.embeddings.npz` with the images.")
    add("")
    add("## Photos measured")
    add("")
    add("OCR off, 1 thread, after the warmup pass. `other` is crop, stick-band, histogram, and counting work.")
    add("")
    if baseline:
        add("| Photo | Pixels | File | Boxes | Wall | CPU | Detect | Embed | Other |")
        add("|---|---:|---:|---:|---:|---:|---:|---:|---:|")
        for row in sorted(ok_rows(baseline), key=lambda item: -item["wall_s"]):
            pixels = f"{row['width']}×{row['height']}" if row.get("width") else ""
            add(
                f"| {row['name']} | {pixels} | {fmt_bytes(row.get('bytes'))} | {row['detections']} | "
                f"{fmt_s(row['wall_s'])} | {fmt_s(row['cpu_s'])} | {fmt_s(row['detect_s'])} | "
                f"{fmt_s(row['embed_s'])} | {fmt_s(row['other_s'])} |"
            )
        failed = [row for row in baseline.get("images", []) if row.get("error")]
        if failed:
            add("")
            add("Failed photos:")
            add("")
            for row in failed:
                add(f"- {row['name']}: {row['error']}")
    add("")
    add("## Method")
    add("")
    add("- Same entry point as `POST /detect`: `run_detect` on the real weights and the repo gallery.")
    add("- One warmup photo is run and discarded, then every photo is timed once. The warmup duration is the first-request row.")
    add("- Stage times are wrappers around the functions the pipeline already calls. They sit inside the wall-clock measurement.")
    add("- RSS after load is a spot sample. Peak RSS is `ru_maxrss` for that process, which includes model load.")
    add("- Each thread setting is a fresh process. OpenMP reads its thread count at first native import, so it cannot be changed later in-process.")
    add("")
    return "\n".join(lines)


def sizing_block(add, payload: dict, summary: dict) -> None:
    size = summary.get("sizing") or {}
    options = summary.get("thread_options") or []
    baseline_option = next((option for option in options if option["threads"] == 1), options[0] if options else None)
    ram = size.get("recommend_ram_gib")
    vcpu = size.get("recommend_vcpu") or 1
    per_min = (baseline_option or {}).get("per_minute")
    ultra = summary.get("ultralytics_num_threads")
    add(
        f"One replica, OCR off, `STOCK_NUM_THREADS` left at 1: **{fmt_gib(ram)} RAM** and **{vcpu} vCPU**. "
        f"That replica does about **{fmt_rate(per_min)} photos/minute** at the measured mean."
    )
    add("")
    add(
        f"The {vcpu} vCPU figure is the measured CPU-seconds per wall-second, rounded up "
        f"({fmt_cores(summary.get('effective_cores'))}). "
        f"Ultralytics still starts {ultra if ultra is not None else 'min(8, cpu_count - 1)'} PyTorch threads on CPU predict, "
        "so a limit of 1 vCPU will not reproduce these times. "
        "A second replica needs the same RAM and CPU again."
    )
    add("")
    if size.get("memory_pressure"):
        add(
            f"Peak RSS was {fmt_bytes(size.get('peak_rss_bytes'))} on a machine with "
            f"{fmt_bytes(size.get('host_ram_bytes'))}. The host was tight on memory, so these times may include swapping. "
            "Re-run the script on the deploy host before locking the numbers in."
        )
        add("")
    add(
        f"The {fmt_gib(ram)} figure is 1.5× the peak RSS with OCR off "
        f"({fmt_bytes(size.get('peak_rss_bytes'))}), rounded up to the next 0.5 GiB. "
        "Set the container memory limit there."
    )
    ocr = summary.get("ocr")
    if ocr and ocr.get("mean_delta_s") is not None:
        add("")
        add(
            f"Turning OCR on added {fmt_s(ocr['mean_delta_s'])} per photo on the sample subset and raised peak RSS to "
            f"{fmt_bytes(ocr.get('maxrss'))}. Leave it off unless the wordmark check is required. "
            f"If it is required, size RAM from that peak ({fmt_gib(recommend_gib(ocr.get('maxrss') or 0))}), not from the baseline."
        )


def sweep_note(add, options: list[dict]) -> None:
    base = next((option for option in options if option["threads"] == 1), None)
    if base is None or len(options) < 2:
        return
    for option in options:
        if option["threads"] == 1 or not option.get("wall_mean_s") or not base.get("wall_mean_s"):
            continue
        latency_ratio = option["wall_mean_s"] / base["wall_mean_s"]
        same_pool = option.get("torch_intra_op") == base.get("torch_intra_op")
        if same_pool:
            add(
                f"`STOCK_NUM_THREADS={option['threads']}` mean latency was {latency_ratio:.2f}× the 1-thread run, "
                f"and both processes used {option.get('torch_intra_op')} PyTorch intra-op threads once predict started. "
                "Changing the env var did not change the pool Ultralytics installs."
            )
        else:
            add(
                f"`STOCK_NUM_THREADS={option['threads']}` mean latency was {latency_ratio:.2f}× the 1-thread run "
                f"(PyTorch intra-op {option.get('torch_intra_op')} vs {base.get('torch_intra_op')})."
            )


def fmt_bytes(n: int | None) -> str:
    if n is None:
        return "n/a"
    value = float(n)
    for unit in ("B", "KiB", "MiB", "GiB", "TiB"):
        if value < 1024 or unit == "TiB":
            if unit == "B":
                return f"{int(value)} {unit}"
            return f"{value:.1f} {unit}"
        value /= 1024
    return f"{value:.1f} TiB"


def fmt_gib(n: float | None) -> str:
    if not n:
        return "n/a"
    text = f"{n:.1f}".rstrip("0").rstrip(".")
    return f"{text} GiB"


def fmt_s(n: float | None) -> str:
    if n is None:
        return "n/a"
    if abs(n) < 10:
        return f"{n:.2f} s"
    return f"{n:.1f} s"


def fmt_rate(n: float | None) -> str:
    if n is None:
        return "n/a"
    return f"{n:.1f}"


def fmt_cores(n: float | None) -> str:
    if n is None:
        return "n/a"
    return f"{n:.2f} cores"


def fmt_num(n: float | None) -> str:
    if n is None:
        return "n/a"
    return f"{n:.1f}"


def fmt_share(part: float | None, whole: float | None) -> str:
    if part is None or not whole:
        return "n/a"
    return f"{100 * part / whole:.0f}%"


def fmt_corr(n: float | None) -> str:
    if n is None:
        return "n/a"
    return f"{n:.2f}"


def fmt_count(n: int | None) -> str:
    if n is None:
        return "n/a"
    if n >= 1_000_000:
        return f"{n / 1_000_000:.1f}M"
    if n >= 1_000:
        return f"{n / 1_000:.1f}k"
    return str(n)


def publish(payload: dict, pdf_path: Path, json_path: Path) -> None:
    pdf_path.parent.mkdir(parents=True, exist_ok=True)
    json_path.parent.mkdir(parents=True, exist_ok=True)
    html_path = pdf_path.with_suffix(".html")
    md_path = pdf_path.with_suffix(".md")
    json_path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    md_path.write_text(render_markdown(payload), encoding="utf-8")
    html_path.write_text(render_html(payload), encoding="utf-8")
    try:
        chrome_pdf(html_path, pdf_path)
    except (OSError, subprocess.CalledProcessError) as exc:
        raise SystemExit(f"Chrome could not write the PDF ({exc}). The HTML report is at {html_path}") from exc
    print(f"wrote {pdf_path}", file=sys.stderr)
    print(f"wrote {html_path}", file=sys.stderr)
    print(f"wrote {md_path}", file=sys.stderr)
    print(f"wrote {json_path}", file=sys.stderr)


def find_chrome() -> Path | None:
    for path in CHROME_CANDIDATES:
        if path.is_file():
            return path
    for name in ("google-chrome", "google-chrome-stable", "chromium", "chromium-browser"):
        found = shutil.which(name)
        if found:
            return Path(found)
    return None


def chrome_pdf(html_path: Path, pdf_path: Path) -> None:
    chrome = find_chrome()
    if chrome is None:
        raise OSError("Google Chrome was not found")
    pdf_path.parent.mkdir(parents=True, exist_ok=True)
    tmp = pdf_path.with_suffix(".pdf.part")
    tmp.unlink(missing_ok=True)
    cmd = [
        str(chrome),
        "--headless=new",
        "--disable-gpu",
        "--no-pdf-header-footer",
        "--no-first-run",
        "--disable-extensions",
        f"--print-to-pdf={tmp}",
        "--virtual-time-budget=30000",
        html_path.resolve().as_uri(),
    ]
    subprocess.run(cmd, check=True, stdout=subprocess.DEVNULL, stderr=subprocess.PIPE, text=True)
    tmp.replace(pdf_path)


def render_html(payload: dict) -> str:
    summary = payload.get("summary") or {}
    sizing = summary.get("sizing") or {}
    baseline = run_by_label(payload, "baseline") or {}
    host = payload.get("host") or {}
    stages = summary.get("stages") or {}
    wall = summary.get("wall_mean_s") or 0
    e = html.escape

    def seg_bar(parts: list[tuple[str, float]], outer: float | None = None) -> str:
        total = sum(max(0.0, value) for _, value in parts) or 1.0
        spans = []
        for name, value in parts:
            width = 100 * max(0.0, value) / total
            if width <= 0:
                continue
            spans.append(f'<span class="seg {name}" style="width:{width:.2f}%"></span>')
        inner = f'<div class="bar">{"".join(spans)}</div>'
        if outer is None:
            return inner
        return f'<div class="track"><div class="bar" style="width:{max(outer, 1.5):.2f}%">{"".join(spans)}</div></div>'

    stage_parts = [
        ("detect", stages.get("detect_s") or 0),
        ("embed", stages.get("embed_s") or 0),
        ("ocr", stages.get("ocr_s") or 0),
        ("rest", (stages.get("orient_s") or 0) + (stages.get("match_s") or 0) + max(0.0, stages.get("other_s") or 0)),
    ]
    rows = sorted(ok_rows(baseline), key=lambda row: -row["wall_s"])
    slowest = max((row["wall_s"] for row in rows), default=1) or 1
    photo_rows = []
    for row in rows:
        parts = [
            ("detect", row["detect_s"]),
            ("embed", row["embed_s"]),
            ("ocr", row["ocr_s"]),
            ("rest", row["orient_s"] + row["match_s"] + max(0.0, row["other_s"])),
        ]
        photo_rows.append(
            "<div class='prow'>"
            f"<div class='pname' title='{e(row['name'])}'>{e(row['name'])}</div>"
            f"{seg_bar(parts, 100 * row['wall_s'] / slowest)}"
            f"<div class='ptime'>{e(fmt_s(row['wall_s']))}</div>"
            f"<div class='pboxes'>{row['detections']}</div>"
            "</div>"
        )

    stat_bits = [
        ("Ready", fmt_s(summary.get("ready_s"))),
        ("First photo", fmt_s(summary.get("first_request_s"))),
        ("Mean", fmt_s(summary.get("wall_mean_s"))),
        ("p50", fmt_s(summary.get("wall_p50_s"))),
        ("p95", fmt_s(summary.get("wall_p95_s"))),
        ("Slowest", fmt_s(summary.get("wall_max_s"))),
    ]
    stats = "".join(
        f"<div class='stat'><div class='label'>{e(label)}</div><div class='value'>{e(value)}</div></div>"
        for label, value in stat_bits
    )

    examples = sizing.get("examples") or []
    replica_cards = "".join(
        "<div class='replica'>"
        f"<div class='value'>{item['replicas_at_p95']}</div>"
        f"<div class='label'>{item['photos_per_minute']} photos/min</div>"
        "</div>"
        for item in examples
    )

    options = summary.get("thread_options") or []
    thread_cards = []
    for option in options:
        thread_cards.append(
            "<div class='tcard'>"
            f"<div class='label'>STOCK_NUM_THREADS={option['threads']}</div>"
            f"<div class='value'>{e(fmt_s(option.get('wall_mean_s')))}</div>"
            f"<p>{e(fmt_rate(option.get('per_minute')))} photos/min · "
            f"PyTorch threads during predict: {option.get('torch_intra_op')}</p>"
            "</div>"
        )

    mem_scale = host.get("ram_bytes") or sizing.get("peak_rss_bytes") or 1
    mem_rows_src = [
        ("After model load", summary.get("rss_after_load")),
        ("After the timed photos", baseline.get("rss_after_requests")),
        ("Peak, OCR off", summary.get("maxrss")),
        ("Recommended limit", int((sizing.get("recommend_ram_gib") or 0) * 1024**3)),
    ]
    ocr = summary.get("ocr")
    if ocr:
        mem_rows_src.append(("Peak, OCR on", ocr.get("maxrss")))
    if host.get("ram_bytes"):
        mem_rows_src.append(("This machine", host.get("ram_bytes")))
    mem_rows = []
    for label, amount in mem_rows_src:
        if not amount:
            continue
        width = min(100, 100 * amount / mem_scale)
        mem_rows.append(
            "<div class='mrow'>"
            f"<div class='mlabel'>{e(label)}</div>"
            f"<div class='track'><div class='bar mem' style='width:{width:.1f}%'></div></div>"
            f"<div class='ptime'>{e(fmt_bytes(amount))}</div>"
            "</div>"
        )

    ocr_block = ""
    if ocr and ocr.get("images"):
        ocr_lines = []
        for row in ocr["images"]:
            ocr_lines.append(
                "<tr>"
                f"<td>{e(row['name'])}</td>"
                f"<td class='num'>{e(fmt_s(row['base_wall_s']))}</td>"
                f"<td class='num'>{e(fmt_s(row['wall_s']))}</td>"
                f"<td class='num'>{e(fmt_s(row['delta_s']))}</td>"
                f"<td class='num'>{row['ocr_calls']}</td>"
                "</tr>"
            )
        ocr_block = f"""
        <section class="panel">
          <h2>Wordmark OCR</h2>
          <p class="note">STOCK_OCR defaults to off. On {ocr.get("photos")} photos it added
          {e(fmt_s(ocr.get("mean_delta_s")))} of wall time and raised peak RSS to {e(fmt_bytes(ocr.get("maxrss")))}.
          PaddleOCR itself accounted for {e(fmt_s(ocr.get("mean_ocr_s")))}. Loading it took {e(fmt_s(ocr.get("load_ocr_s")))}.</p>
          <table>
            <thead><tr><th>Photo</th><th class="num">OCR off</th><th class="num">OCR on</th><th class="num">Added</th><th class="num">Calls</th></tr></thead>
            <tbody>{"".join(ocr_lines)}</tbody>
          </table>
        </section>
        """
    elif payload.get("ocr_error"):
        ocr_block = f"<section class='panel'><h2>Wordmark OCR</h2><pre>{e(payload['ocr_error'])}</pre></section>"

    disk_lines = []
    for row in (payload.get("disk") or {}).get("artifacts") or []:
        missing = "" if row.get("present", True) else " — not on this machine"
        disk_lines.append(
            "<tr>"
            f"<td>{e(row['name'])}{e(missing)}</td>"
            f"<td>{e(row['role'])}</td>"
            f"<td class='num'>{e(fmt_bytes(row.get('bytes')))}</td>"
            "</tr>"
        )
    package_note = fmt_bytes((payload.get("disk") or {}).get("python_packages_bytes"))

    models = baseline.get("models") or {}
    groups = models.get("siglip_groups") or {}
    group_txt = ", ".join(f"{e(name)} {e(fmt_count(count))}" for name, count in groups.items())
    gallery = baseline.get("gallery") or {}
    cfg = baseline.get("config") or {}
    git = payload.get("git") or {}
    dirty = " · uncommitted changes" if git.get("dirty") else ""
    code = f"{git.get('short', '')} on {git.get('branch', '')}{dirty}" if git else ""
    perf = host.get("performance_cores")
    eff = host.get("efficiency_cores")
    core_note = f"{host.get('logical_cpus')} logical cores"
    if perf or eff:
        core_note += f", {perf or 0} performance + {eff or 0} efficiency"
    cpu_name = host.get("cpu") or host.get("machine") or ""
    corr = summary.get("correlations") or {}
    ultra = summary.get("ultralytics_num_threads")
    during = summary.get("torch_threads_during_requests") or {}
    after = summary.get("torch_threads_after_load") or {}
    cpu_sentence = (
        f"Startup sets PyTorch to {after.get('intra_op', 1)} thread, then Ultralytics raises it to "
        f"{ultra if ultra is not None else 'min(8, cpu_count − 1)'} on every CPU predict "
        f"(measured {during.get('intra_op', 'n/a')} while photos were running). "
        "A 1 vCPU cap will miss these times."
    )
    if len(options) > 1 and len({opt.get("torch_intra_op") for opt in options}) == 1:
        asked = " and ".join(str(opt["threads"]) for opt in options)
        cpu_sentence += f" Trying {asked} threads ended on that same pool."
    ocr_ram = ""
    if ocr and ocr.get("maxrss"):
        ocr_ram = (
            f" With OCR on, use {e(fmt_gib(recommend_gib(ocr['maxrss'])))} "
            f"from the {e(fmt_bytes(ocr['maxrss']))} peak."
        )
    memory_sentence = (
        f"Set the limit to <strong>{e(fmt_gib(sizing.get('recommend_ram_gib')))}</strong>, "
        f"1.5× the {e(fmt_bytes(summary.get('maxrss')))} peak RSS, rounded up.{ocr_ram}"
    )

    body = f"""
    <header class="hero">
      <p class="eyebrow">Processing capacity · POST /detect</p>
      <h1>Stock material detection</h1>
      <p>What one replica costs in production. Measured {e(pretty_when(payload.get("generated_at")))}
      on {e(host.get("hostname") or "this machine")}.</p>
      <p class="sub">{e(cpu_name)} · {e(core_note)} · {e(fmt_bytes(host.get("ram_bytes")))} RAM
      · Python {e(str(host.get("python") or ""))} · {e(code)}</p>
    </header>

    <section class="kpis">
      <div class="kpi"><div class="label">RAM per replica</div><div class="value">{e(fmt_gib(sizing.get("recommend_ram_gib")))}</div><div class="hint">OCR off</div></div>
      <div class="kpi"><div class="label">vCPU per replica</div><div class="value">{sizing.get("recommend_vcpu") or "n/a"}</div><div class="hint">{e(fmt_cores(summary.get("effective_cores")))} measured</div></div>
      <div class="kpi"><div class="label">Photos / minute</div><div class="value">{e(fmt_rate(summary.get("per_minute_at_mean")))}</div><div class="hint">{e(fmt_rate(summary.get("per_minute_at_p95")))} at p95</div></div>
      <div class="kpi"><div class="label">Typical request</div><div class="value">{e(fmt_s(summary.get("wall_p50_s")))}</div><div class="hint">p95 {e(fmt_s(summary.get("wall_p95_s")))}</div></div>
    </section>

    <section class="calls">
      <article class="callout">
        <h2>Container</h2>
        <p>One uvicorn process, <strong>--workers 1</strong>. The API holds one lock around the whole pipeline, so a second photo waits. Another worker is a second copy of the weights. To run two photos at once, run two containers.</p>
      </article>
      <article class="callout">
        <h2>CPU limit</h2>
        <p>Give the container <strong>{sizing.get("recommend_vcpu") or "n/a"} vCPU</strong>. {cpu_sentence}</p>
      </article>
      <article class="callout">
        <h2>Memory limit</h2>
        <p>{memory_sentence}</p>
      </article>
    </section>

    <section class="panel">
      <h2>Where a request spends its time</h2>
      <p class="note">Mean of {summary.get("photos", 0)} photos, OCR off. The detector resizes every photo to {cfg.get("detector_image_size", 960)} px, so a larger upload does not make YOLOE slower. SigLIP embeds each box in four quarter-turns, one call per box.</p>
      {seg_bar(stage_parts)}
      <div class="legend">
        <span><i class="swatch detect"></i>YOLOE {e(fmt_s(stages.get("detect_s")))} · {e(fmt_share(stages.get("detect_s"), wall))}</span>
        <span><i class="swatch embed"></i>SigLIP {e(fmt_s(stages.get("embed_s")))} · {e(fmt_share(stages.get("embed_s"), wall))}</span>
        <span><i class="swatch rest"></i>Decode, crops, match {e(fmt_s(stage_parts[-1][1]))}</span>
      </div>
      <div class="stats">{stats}</div>
      <p class="note">Request time tracks box count (correlation {e(fmt_corr(corr.get("wall_vs_detections")))}). It barely tracks megapixels ({e(fmt_corr(corr.get("wall_vs_megapixels")))}). Mean {e(fmt_num(summary.get("mean_detections")))} boxes per photo.</p>
    </section>

    <section class="panel">
      <h2>Replicas to hold the p95</h2>
      <p class="note">A load balancer spreads photos evenly and each replica stays within the measured p95 ({e(fmt_s(summary.get("wall_p95_s")))}), with no queue. Startup probe {sizing.get("startup_probe_s", "n/a")} s. Proxy timeout {sizing.get("request_timeout_s", "n/a")} s. There is no GPU path; the detector is fixed to CPU.</p>
      <div class="replicas">{replica_cards}</div>
    </section>

    <section class="panel photos">
      <h2>Every photo</h2>
      <p class="note">Bar length is wall time. Colour is the same split as above. The first request after load ({e(fmt_s(summary.get("first_request_s")))}) is not in this list.</p>
      <div class="phead"><span>Photo</span><span>Detect · SigLIP</span><span class="ptime">Wall</span><span class="pboxes">Boxes</span></div>
      {"".join(photo_rows)}
    </section>

    <section class="panel">
      <h2>Thread setting</h2>
      <p class="note">Each column is a fresh process. OpenCV stayed at {during.get("opencv", "n/a")} thread. Both predicts ran with the same PyTorch intra-op pool, so the means match.</p>
      <div class="threads">{"".join(thread_cards)}</div>
    </section>

    <section class="panel">
      <h2>Resident memory</h2>
      <p class="note">Bars are scaled to this machine’s RAM. Peak RSS is the OS high-water mark and includes model load. Queued uploads sit on top of this: each waiting request holds up to 25 MiB before it takes the lock.</p>
      {"".join(mem_rows)}
    </section>

    {ocr_block}

    <section class="panel split">
      <h2>What to bake into the image</h2>
      <p class="note">Weights download on first use. Bake them into the image so a new container does not download during the readiness check. Gallery: {gallery.get("skus", "n/a")} SKUs, {gallery.get("references", "n/a")} reference crops, embed dim {gallery.get("embed_dim", "n/a")}. Backend <code>{e(str(baseline.get("backend") or ""))}</code>. Python packages on disk: {e(package_note)}.</p>
      <table>
        <thead><tr><th>Artifact</th><th>Role</th><th class="num">Size</th></tr></thead>
        <tbody>{"".join(disk_lines)}</tbody>
      </table>
      <p class="note model-note">YOLOE {e(fmt_count(models.get("yolo_params")))} parameters ({e(fmt_bytes(models.get("yolo_bytes")))}). SigLIP 2 {e(fmt_count(models.get("siglip_params")))} parameters ({e(fmt_bytes(models.get("siglip_bytes")))}); requests use the image tower, and the text tower stays loaded ({group_txt}). Ship <code>gallery/.embeddings.npz</code> with the images.</p>
    </section>
    """
    return HTML_DOCUMENT.replace("<!--BODY-->", body)


def pretty_when(value: str | None) -> str:
    if not value:
        return ""
    try:
        moment = datetime.fromisoformat(value)
    except ValueError:
        return value
    return moment.strftime("%d %B %Y, %H:%M")


HTML_DOCUMENT = """<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <title>Processing capacity — stock material detection</title>
  <style>
    :root {
      --ink: #1c2430;
      --muted: #5c6b7e;
      --line: #d5deea;
      --bg: #eef2f6;
      --card: #ffffff;
      --brand: #12325c;
      --accent: #2b6cb0;
      --embed: #12325c;
      --rest: #c5d0de;
      --amber: #9a6700;
      --amber-bg: #fff6dd;
    }
    * { box-sizing: border-box; }
    html, body { margin: 0; padding: 0; }
    body {
      font-family: "Segoe UI", "Helvetica Neue", Arial, sans-serif;
      color: var(--ink);
      background: var(--bg);
      line-height: 1.45;
    }
    .page { max-width: 920px; margin: 0 auto; padding: 28px 22px 48px; }
    header.hero {
      background: var(--brand);
      color: #fff;
      border-radius: 16px;
      padding: 26px 28px 22px;
      margin-bottom: 16px;
    }
    .eyebrow {
      margin: 0 0 6px;
      letter-spacing: .14em;
      text-transform: uppercase;
      font-size: 11px;
      color: #b9cbe4;
    }
    header.hero h1 { margin: 0 0 8px; font-size: 30px; letter-spacing: -0.03em; }
    header.hero p { margin: 0; color: #d7e3f4; max-width: 68ch; }
    header.hero .sub { margin-top: 10px; font-size: 12px; color: #b9cbe4; }
    .kpis {
      display: grid;
      grid-template-columns: repeat(4, 1fr);
      gap: 10px;
      margin-bottom: 12px;
    }
    .kpi, .panel, .callout, .tcard, .replica {
      background: var(--card);
      border: 1px solid var(--line);
      border-radius: 12px;
    }
    .kpi { padding: 12px 14px 10px; }
    .label { color: var(--muted); font-size: 11px; text-transform: uppercase; letter-spacing: .05em; }
    .kpi .value { font-size: 28px; font-weight: 700; letter-spacing: -0.03em; margin-top: 2px; color: var(--brand); }
    .hint { color: var(--muted); font-size: 12px; }
    .calls {
      display: grid;
      grid-template-columns: 1fr 1fr 1fr;
      gap: 10px;
      margin-bottom: 12px;
    }
    .callout { padding: 14px 14px 12px; }
    .callout h2, .panel h2 { margin: 0 0 6px; font-size: 15px; color: var(--brand); }
    .callout p, .note { margin: 0; color: var(--muted); font-size: 13px; }
    .panel { padding: 16px 18px 14px; margin-bottom: 12px; }
    .panel h2 { font-size: 18px; }
    .note { margin: 0 0 12px; }
    .bar {
      display: flex;
      height: 16px;
      border-radius: 8px;
      overflow: hidden;
      background: #e7edf4;
    }
    .track { background: #f4f7fb; border: 1px solid #e3ebf3; border-radius: 8px; overflow: hidden; }
    .track .bar { background: transparent; border-radius: 0; height: 14px; }
    .seg { display: block; height: 100%; }
    .seg.detect, .swatch.detect, .bar.mem { background: var(--accent); }
    .seg.embed, .swatch.embed { background: var(--embed); }
    .seg.ocr, .swatch.ocr { background: #c05621; }
    .seg.rest, .swatch.rest { background: #7d92a8; }
    .legend { display: flex; flex-wrap: wrap; gap: 14px; margin: 8px 0 14px; font-size: 12px; color: var(--muted); }
    .legend span { display: inline-flex; align-items: center; gap: 6px; }
    .swatch { width: 10px; height: 10px; border-radius: 2px; display: inline-block; }
    .stats { display: grid; grid-template-columns: repeat(6, 1fr); gap: 8px; margin-bottom: 10px; }
    .stat { background: #f6f8fb; border-radius: 8px; padding: 8px 8px 6px; }
    .stat .value { font-size: 16px; font-weight: 700; color: var(--ink); font-variant-numeric: tabular-nums; }
    .replicas, .threads { display: grid; gap: 10px; }
    .replicas { grid-template-columns: repeat(4, 1fr); }
    .threads { grid-template-columns: 1fr 1fr; }
    .replica, .tcard { padding: 12px 14px; }
    .replica .value, .tcard .value { font-size: 26px; font-weight: 700; color: var(--brand); letter-spacing: -0.03em; }
    .tcard p { margin: 4px 0 0; color: var(--muted); font-size: 12px; }
    .phead, .prow, .mrow {
      display: grid;
      grid-template-columns: 210px 1fr 64px 42px;
      gap: 8px;
      align-items: center;
    }
    .phead { font-size: 11px; text-transform: uppercase; letter-spacing: .04em; color: var(--muted); margin-bottom: 4px; }
    .prow, .mrow { margin: 3px 0; break-inside: avoid; page-break-inside: avoid; }
    .pname, .mlabel {
      font-size: 12px;
      white-space: nowrap;
      overflow: hidden;
      text-overflow: ellipsis;
    }
    .ptime, .pboxes { font-size: 12px; text-align: right; font-variant-numeric: tabular-nums; }
    .mrow { grid-template-columns: 160px 1fr 72px; }
    table { width: 100%; border-collapse: collapse; font-size: 12.5px; }
    th, td { text-align: left; padding: 6px 6px; border-bottom: 1px solid var(--line); vertical-align: top; }
    th { font-size: 10px; text-transform: uppercase; letter-spacing: .04em; color: var(--muted); }
    td.num, th.num { text-align: right; font-variant-numeric: tabular-nums; }
    code { font-family: ui-monospace, "SF Mono", Menlo, Consolas, monospace; font-size: 0.92em; }
    .model-note { margin-top: 10px; }
    footer { color: var(--muted); font-size: 11px; margin-top: 8px; }
    @media print {
      body { background: #fff; }
      .page { max-width: none; padding: 0; }
      header.hero, .kpi, .panel, .callout, .tcard, .replica, .seg, .bar.mem, .stat, .swatch {
        -webkit-print-color-adjust: exact;
        print-color-adjust: exact;
      }
      .panel, .callout, .kpi { break-inside: avoid; page-break-inside: avoid; }
      .photos, .split { break-inside: auto; page-break-inside: auto; }
      tr { break-inside: avoid; page-break-inside: avoid; }
    }
    @page { size: A4; margin: 12mm 12mm 14mm; }
  </style>
</head>
<body>
  <div class="page">
<!--BODY-->
  </div>
</body>
</html>
"""


if __name__ == "__main__":
    main()
