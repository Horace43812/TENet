import argparse
from pathlib import Path

from torchvision import datasets


MIRROR_VARIANTS = [
    [
        (
            "https://github.com/zalandoresearch/fashion-mnist/raw/master/data/fashion/train-images-idx3-ubyte.gz",
            "f68b3c2dcbeaaa9fbdd348bbdeb45c12",
        ),
        (
            "https://github.com/zalandoresearch/fashion-mnist/raw/master/data/fashion/train-labels-idx1-ubyte.gz",
            "f17e45b128099976906a5758510dd100",
        ),
        (
            "https://github.com/zalandoresearch/fashion-mnist/raw/master/data/fashion/t10k-images-idx3-ubyte.gz",
            "bef4ecab320f06d8554227c12da6a029",
        ),
        (
            "https://github.com/zalandoresearch/fashion-mnist/raw/master/data/fashion/t10k-labels-idx1-ubyte.gz",
            "bb300cfdad3c16e7a124157e65021045",
        ),
    ],
    [
        (
            "https://raw.githubusercontent.com/zalandoresearch/fashion-mnist/master/data/fashion/train-images-idx3-ubyte.gz",
            "f68b3c2dcbeaaa9fbdd348bbdeb45c12",
        ),
        (
            "https://raw.githubusercontent.com/zalandoresearch/fashion-mnist/master/data/fashion/train-labels-idx1-ubyte.gz",
            "f17e45b128099976906a5758510dd100",
        ),
        (
            "https://raw.githubusercontent.com/zalandoresearch/fashion-mnist/master/data/fashion/t10k-images-idx3-ubyte.gz",
            "bef4ecab320f06d8554227c12da6a029",
        ),
        (
            "https://raw.githubusercontent.com/zalandoresearch/fashion-mnist/master/data/fashion/t10k-labels-idx1-ubyte.gz",
            "bb300cfdad3c16e7a124157e65021045",
        ),
    ],
    [
        (
            "https://cdn.jsdelivr.net/gh/zalandoresearch/fashion-mnist@master/data/fashion/train-images-idx3-ubyte.gz",
            "f68b3c2dcbeaaa9fbdd348bbdeb45c12",
        ),
        (
            "https://cdn.jsdelivr.net/gh/zalandoresearch/fashion-mnist@master/data/fashion/train-labels-idx1-ubyte.gz",
            "f17e45b128099976906a5758510dd100",
        ),
        (
            "https://cdn.jsdelivr.net/gh/zalandoresearch/fashion-mnist@master/data/fashion/t10k-images-idx3-ubyte.gz",
            "bef4ecab320f06d8554227c12da6a029",
        ),
        (
            "https://cdn.jsdelivr.net/gh/zalandoresearch/fashion-mnist@master/data/fashion/t10k-labels-idx1-ubyte.gz",
            "bb300cfdad3c16e7a124157e65021045",
        ),
    ],
]


def ensure_relative_root(root: str) -> Path:
    root_path = Path(root)
    if root_path.is_absolute():
        raise ValueError(
            "--root 必须是相对路径，例如 './data' 或 '../datasets/fashion_mnist'。"
        )
    root_path.mkdir(parents=True, exist_ok=True)
    return root_path


def download_with_fallback(root: Path) -> None:
    errors = []
    for idx, resources in enumerate(MIRROR_VARIANTS, start=1):
        datasets.FashionMNIST.resources = resources
        try:
            print(f"[Mirror {idx}] 下载训练集...")
            train_set = datasets.FashionMNIST(root=str(root), train=True, download=True)
            print(f"[Mirror {idx}] 下载测试集...")
            test_set = datasets.FashionMNIST(root=str(root), train=False, download=True)
            print(
                f"下载完成: train={len(train_set)}, test={len(test_set)}, root={root.as_posix()}"
            )
            return
        except Exception as exc:
            errors.append(f"Mirror {idx}: {exc}")

    detail = "\n".join(errors)
    raise RuntimeError(f"所有镜像下载失败。\n{detail}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="下载 FashionMNIST（仅相对路径，内置多镜像重试）"
    )
    parser.add_argument(
        "--root",
        type=str,
        default="./data",
        help="数据集保存目录（相对路径），默认: ./data",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    root = ensure_relative_root(args.root)
    download_with_fallback(root)


if __name__ == "__main__":
    main()
