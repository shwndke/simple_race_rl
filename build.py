import subprocess
import argparse
from pathlib import Path


def main() -> None:
    root = Path(__file__).resolve().parent
    build = root / "build"
    build.mkdir(exist_ok=True)
    parser = argparse.ArgumentParser()
    parser.add_argument("--name", default="racing")
    parser.add_argument("--source", default="racing.c")
    args = parser.parse_args()
    assert args.name.isidentifier(), "Use an identifier for the output name"

    result = subprocess.run(
        [
            "cl",
            "/nologo",
            "/O2",
            "/LD",
            str(root / args.source),
            f"/Fo{build / (args.name + '.obj')}",
            f"/Fe{build / (args.name + '.dll')}",
            "/link",
            f"/IMPLIB:{build / (args.name + '.lib')}",
        ],
        cwd=root,
    )
    assert result.returncode == 0


if __name__ == "__main__":
    main()
