import argparse
import json
import logging
import re
from pathlib import Path

import numpy as np

from solid_to_triangles2 import process_main


ADV_FACE_RE = re.compile(
    r"^\s*#\d+\s*=\s*ADVANCED_FACE\(\s*'([^']*)'\s*,",
    re.IGNORECASE,
)


def extract_face_labels_from_step(step_path: Path):
    """
    从 STEP 文件中提取 ADVANCED_FACE 的第一个元素作为面标签。
    示例: #17 = ADVANCED_FACE('24',(#18,#193),#32,.F.);
    返回: [24, ...] （按文件中出现顺序）
    """
    labels = []
    with step_path.open("r", encoding="utf-8", errors="ignore") as f:
        for line in f:
            m = ADV_FACE_RE.match(line)
            if not m:
                continue
            raw = m.group(1).strip()
            # 优先转 int；若失败则尝试 float->int；再失败用 -1 占位
            try:
                v = int(raw)
            except ValueError:
                try:
                    v = int(float(raw))
                except ValueError:
                    v = -1
            labels.append(v)
    return labels


def collect_step_files(root: Path):
    return sorted([*root.rglob("*.step"), *root.rglob("*.stp")])


def main():
    parser = argparse.ArgumentParser("Generate MFCAD BRT topology and face labels from STEP")
    parser.add_argument("data_path", type=str, help="MFCAD 数据根目录（包含 STEP）")
    parser.add_argument("output_path", type=str, help="输出目录（BRT 拓扑 + labels）")
    parser.add_argument("--method", type=int, default=10)
    parser.add_argument("--process_num", type=int, default=30)
    parser.add_argument("--dataset", type=str, default="mfcad")
    parser.add_argument("--target", type=str, default="brt")
    parser.add_argument(
        "--label_subdir",
        type=str,
        default="labels",
        help="输出目录下保存面标签的子目录名",
    )
    args = parser.parse_args()

    log_dir = Path("logs")
    log_dir.mkdir(parents=True, exist_ok=True)
    logging.basicConfig(
        filename=str(log_dir / "gen_mfcad_topo.log"),
        filemode="w",
        format="%(asctime)s :: %(levelname)-8s :: %(message)s",
        level=logging.INFO,
    )

    data_root = Path(args.data_path)
    out_root = Path(args.output_path)
    label_root = out_root / args.label_subdir
    label_root.mkdir(parents=True, exist_ok=True)

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

    step_files = collect_step_files(data_root)
    logging.info("Found %d STEP files", len(step_files))

    manifest = {}
    ok, bad = 0, 0

    for step_file in step_files:
        rel = step_file.relative_to(data_root)
        label_file = (label_root / rel).with_suffix(".txt")
        label_file.parent.mkdir(parents=True, exist_ok=True)

        labels = extract_face_labels_from_step(step_file)
        if len(labels) == 0:
            logging.warning("No ADVANCED_FACE label found: %s", step_file)
            bad += 1
            continue

        np.savetxt(label_file, np.asarray(labels, dtype=np.int64), fmt="%d")
        manifest[str(rel)] = str(label_file.relative_to(out_root))
        ok += 1

    manifest_path = out_root / "mfcad_label_manifest.json"
    with manifest_path.open("w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2, ensure_ascii=False)

    logging.info("Done. label_ok=%d, label_failed=%d, manifest=%s", ok, bad, manifest_path)
    print(f"Done. label_ok={ok}, label_failed={bad}")
    print(f"Manifest: {manifest_path}")


if __name__ == "__main__":
    main()