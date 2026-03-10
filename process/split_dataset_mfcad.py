import argparse
import json
from pathlib import Path


SPLITS = ("train", "val", "test")
SPLIT_ALIASES = {
    "train": ("train", "trian"),  # 兼容 triangles/trian
    "val": ("val",),
    "test": ("test",),
}


def collect_by_split(root: Path, split_name: str, pattern: str):
    """
    从 root/<split_alias>/ 下收集文件，返回:
    {sample_id(去后缀文件名): 绝对路径}
    """
    result = {}
    aliases = SPLIT_ALIASES[split_name]
    for alias in aliases:
        split_dir = root / alias
        if not split_dir.exists():
            continue
        for p in split_dir.rglob(pattern):
            sid = p.stem
            # 若重复 sample_id，后者覆盖前者
            result[sid] = str(p)
    return result


def build_items(split_name: str, topo_root: Path, face_root: Path, label_root: Path):
    topo_map = collect_by_split(topo_root, split_name, "*.bin")
    face_map = collect_by_split(face_root, split_name, "*.bin")
    label_map = collect_by_split(label_root, split_name, "*.txt")

    all_ids = sorted(set(topo_map.keys()) | set(face_map.keys()) | set(label_map.keys()))
    items, missing = [], []

    for sid in all_ids:
        topo = topo_map.get(sid)
        face = face_map.get(sid)
        label = label_map.get(sid)

        if topo and face and label:
            items.append({"topo": topo, "face": face, "label": label})
        else:
            missing.append(
                {
                    "id": sid,
                    "topo": bool(topo),
                    "face": bool(face),
                    "label": bool(label),
                }
            )
    return items, missing


def main():
    parser = argparse.ArgumentParser("Generate datasplit.json from split folders")
    parser.add_argument(
        "mfcad_root",
        type=str,
        help="根目录（包含 topo/ triangles/）",
    )
    parser.add_argument(
        "--labels_root",
        type=str,
        default=None,
        help="labels 根目录；默认先找 mfcad_root/labels，不存在则找 mfcad_root 的同级 labels",
    )
    parser.add_argument("--topo_subdir", type=str, default="topo")
    parser.add_argument("--face_subdir", type=str, default="triangles")
    parser.add_argument("--output_file", type=str, default="datasplit.json")
    args = parser.parse_args()

    mfcad_root = Path(args.mfcad_root).resolve()
    topo_root = mfcad_root / args.topo_subdir
    face_root = mfcad_root / args.face_subdir

    if args.labels_root is not None:
        label_root = Path(args.labels_root).resolve()
    else:
        label_root = mfcad_root / "labels"
        if not label_root.exists():
            label_root = mfcad_root.parent / "labels"

    if not topo_root.exists():
        raise FileNotFoundError(f"topo root not found: {topo_root}")
    if not face_root.exists():
        raise FileNotFoundError(f"face root not found: {face_root}")
    if not label_root.exists():
        raise FileNotFoundError(f"label root not found: {label_root}")

    datasplit = {}
    missing_all = {}

    for split in SPLITS:
        items, missing = build_items(split, topo_root, face_root, label_root)
        datasplit[split] = items
        missing_all[split] = missing

    out_file = mfcad_root / args.output_file
    with out_file.open("w", encoding="utf-8") as f:
        json.dump(datasplit, f, indent=2, ensure_ascii=False)

    print(f"Saved: {out_file}")
    for split in SPLITS:
        print(f"{split}: {len(datasplit[split])}, missing: {len(missing_all[split])}")

    if any(len(v) > 0 for v in missing_all.values()):
        miss_file = mfcad_root / "datasplit_missing_ids.json"
        with miss_file.open("w", encoding="utf-8") as f:
            json.dump(missing_all, f, indent=2, ensure_ascii=False)
        print(f"Missing details saved: {miss_file}")


if __name__ == "__main__":
    main()