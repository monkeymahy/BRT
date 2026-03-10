import argparse
import json
import logging
from pathlib import Path

import numpy as np

from solid_to_triangles2 import process_main


def collect_step_files(root: Path):
    return sorted([*root.rglob("*.step"), *root.rglob("*.stp")])


def collect_json_files(root: Path):
    return sorted(root.rglob("*.json"))


def _to_int_label(value):
    try:
        return int(value)
    except (TypeError, ValueError):
        try:
            return int(float(value))
        except (TypeError, ValueError):
            return -1


def load_labels_from_json(json_path: Path):
    with json_path.open("r", encoding="utf-8") as f:
        data = json.load(f)

    if not isinstance(data, list):
        raise ValueError(f"Label json is not a list: {json_path}")

    labels = [_to_int_label(v) for v in data]
    return labels


def build_stem_index(json_files):
    stem_index = {}
    for p in json_files:
        stem_index.setdefault(p.stem, []).append(p)
    return stem_index


def resolve_label_json(step_file: Path, step_root: Path, label_root: Path, stem_index):
    rel = step_file.relative_to(step_root)

    candidate = (label_root / rel).with_suffix(".json")
    if candidate.exists():
        return candidate

    flat_candidate = label_root / f"{step_file.stem}.json"
    if flat_candidate.exists():
        return flat_candidate

    by_stem = stem_index.get(step_file.stem, [])
    if len(by_stem) == 1:
        return by_stem[0]

    return None


def main():
    parser = argparse.ArgumentParser("Generate SFCAD BRT topology and face labels from JSON")
    parser.add_argument("data_path", type=str, help="SFCAD 数据根目录")
    parser.add_argument("output_path", type=str, help="输出目录（BRT 拓扑 + labels）")
    parser.add_argument("--step_subdir", type=str, default="", help="STEP 子目录（默认直接用 data_path）")
    parser.add_argument(
        "--label_subdir",
        type=str,
        default="labels",
        help="输入标签子目录（json），相对 data_path",
    )
    parser.add_argument(
        "--output_label_subdir",
        type=str,
        default="labels",
        help="输出标签子目录（txt），相对 output_path",
    )
    parser.add_argument("--method", type=int, default=10)
    parser.add_argument("--process_num", type=int, default=30)
    parser.add_argument("--dataset", type=str, default="sfcad")
    parser.add_argument("--target", type=str, default="brt")
    args = parser.parse_args()

    log_dir = Path("logs")
    log_dir.mkdir(parents=True, exist_ok=True)
    logging.basicConfig(
        filename=str(log_dir / "gen_sfcad_topo.log"),
        filemode="w",
        format="%(asctime)s :: %(levelname)-8s :: %(message)s",
        level=logging.INFO,
    )

    data_root = Path(args.data_path).resolve()
    out_root = Path(args.output_path).resolve()

    step_root = (data_root / args.step_subdir).resolve() if args.step_subdir else data_root
    label_root = (data_root / args.label_subdir).resolve()
    out_label_root = out_root / args.output_label_subdir
    out_label_root.mkdir(parents=True, exist_ok=True)

    if not step_root.exists():
        raise FileNotFoundError(f"step root not found: {step_root}")
    if not label_root.exists():
        raise FileNotFoundError(f"label root not found: {label_root}")

    logging.info("Start BRT conversion...")
    process_main(
        str(data_root),
        str(out_root),
        method=args.method,
        dataset=args.dataset,
        target=args.target,
        process_num=args.process_num,
    )
    logging.info("BRT conversion done.")

    step_files = collect_step_files(step_root)
    json_files = collect_json_files(label_root)
    stem_index = build_stem_index(json_files)

    logging.info("Found %d STEP files", len(step_files))
    logging.info("Found %d JSON label files", len(json_files))

    manifest = {}
    ok, miss, bad = 0, 0, 0

    for step_file in step_files:
        rel = step_file.relative_to(step_root)
        label_json = resolve_label_json(step_file, step_root, label_root, stem_index)

        if label_json is None:
            logging.warning("No matched label json for STEP: %s", step_file)
            miss += 1
            continue

        try:
            labels = load_labels_from_json(label_json)
        except Exception as exc:
            logging.warning("Failed to parse label json %s: %s", label_json, exc)
            bad += 1
            continue

        if len(labels) == 0:
            logging.warning("Empty label json: %s", label_json)
            bad += 1
            continue

        txt_path = (out_label_root / rel).with_suffix(".txt")
        txt_path.parent.mkdir(parents=True, exist_ok=True)
        np.savetxt(txt_path, np.asarray(labels, dtype=np.int64), fmt="%d")

        manifest[str(rel)] = {
            "json": str(label_json.relative_to(data_root)),
            "txt": str(txt_path.relative_to(out_root)),
        }
        ok += 1

    manifest_path = out_root / "sfcad_label_manifest.json"
    with manifest_path.open("w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2, ensure_ascii=False)

    logging.info(
        "Done. label_ok=%d, label_missing=%d, label_failed=%d, manifest=%s",
        ok,
        miss,
        bad,
        manifest_path,
    )
    print(f"Done. label_ok={ok}, label_missing={miss}, label_failed={bad}")
    print(f"Manifest: {manifest_path}")


if __name__ == "__main__":
    main()
