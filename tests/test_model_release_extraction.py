import ast
from hashlib import sha256
from io import BytesIO
from pathlib import Path
import sys
import tarfile

import pytest


PRODUCTION = Path(__file__).resolve().parents[1] / "production"
sys.path.insert(0, str(PRODUCTION))

from models.extract_release import (  # noqa: E402
    UnsafeModelArchive,
    extract_model_release,
)


def _archive(tmp_path: Path, members: list[tuple[str, bytes, str]]) -> tuple[Path, str]:
    path = tmp_path / "models.tar.gz"
    with tarfile.open(path, "w:gz") as archive:
        for name, payload, kind in members:
            info = tarfile.TarInfo(name)
            if kind == "file":
                info.size = len(payload)
                archive.addfile(info, BytesIO(payload))
            elif kind == "symlink":
                info.type = tarfile.SYMTYPE
                info.linkname = payload.decode()
                archive.addfile(info)
            else:  # pragma: no cover - test helper misuse
                raise ValueError(kind)
    digest = sha256(path.read_bytes()).hexdigest()
    return path, digest


def test_extracts_checksum_pinned_files_under_model_prefix(tmp_path):
    archive, digest = _archive(tmp_path, [
        ("results/professional_tennis/XGBoost/releases/model.json", b"{}", "file"),
    ])

    count = extract_model_release(archive, tmp_path / "checkout", digest)

    assert count == 1
    assert (
        tmp_path / "checkout/results/professional_tennis/XGBoost/releases/model.json"
    ).read_bytes() == b"{}"


@pytest.mark.parametrize("name", [
    "../production/main.py",
    "/tmp/escaped",
    "production/main.py",
    "results/professional_tennis/../../production/main.py",
    "results\\professional_tennis\\model.pth",
])
def test_rejects_paths_outside_model_results(tmp_path, name):
    archive, digest = _archive(tmp_path, [(name, b"bad", "file")])

    with pytest.raises(UnsafeModelArchive):
        extract_model_release(archive, tmp_path / "checkout", digest)


def test_rejects_links(tmp_path):
    archive, digest = _archive(tmp_path, [
        ("results/professional_tennis/model.pth", b"../../production/main.py", "symlink"),
    ])

    with pytest.raises(UnsafeModelArchive, match="regular file or directory"):
        extract_model_release(archive, tmp_path / "checkout", digest)


def test_rejects_duplicate_members(tmp_path):
    name = "results/professional_tennis/model.pth"
    archive, digest = _archive(tmp_path, [
        (name, b"first", "file"),
        (name, b"second", "file"),
    ])

    with pytest.raises(UnsafeModelArchive, match="duplicate archive member"):
        extract_model_release(archive, tmp_path / "checkout", digest)


def test_rejects_wrong_archive_checksum_before_extraction(tmp_path):
    archive, _ = _archive(tmp_path, [
        ("results/professional_tennis/model.pth", b"weights", "file"),
    ])

    with pytest.raises(UnsafeModelArchive, match="SHA-256 mismatch"):
        extract_model_release(archive, tmp_path / "checkout", "0" * 64)
    assert not (tmp_path / "checkout/results").exists()


def test_rejects_existing_file_collision(tmp_path):
    archive, digest = _archive(tmp_path, [
        ("results/professional_tennis/README.md", b"replacement", "file"),
    ])
    checkout = tmp_path / "checkout"
    existing = checkout / "results/professional_tennis/README.md"
    existing.parent.mkdir(parents=True)
    existing.write_text("tracked", encoding="utf-8")

    with pytest.raises(UnsafeModelArchive, match="overwrite existing files"):
        extract_model_release(archive, checkout, digest)
    assert existing.read_text(encoding="utf-8") == "tracked"


def test_all_torch_loads_request_weights_only_explicitly():
    repo = Path(__file__).resolve().parents[1]
    violations: list[str] = []
    for root in (repo / "production", repo / "src"):
        for path in root.rglob("*.py"):
            source = path.read_text(encoding="utf-8")
            if "torch.load" not in source:
                continue
            tree = ast.parse(source, filename=str(path))
            for node in ast.walk(tree):
                if not isinstance(node, ast.Call):
                    continue
                function = node.func
                if not (
                    isinstance(function, ast.Attribute)
                    and function.attr == "load"
                    and isinstance(function.value, ast.Name)
                    and function.value.id == "torch"
                ):
                    continue
                keyword = next(
                    (item for item in node.keywords if item.arg == "weights_only"),
                    None,
                )
                if not (
                    keyword is not None
                    and isinstance(keyword.value, ast.Constant)
                    and keyword.value.value is True
                ):
                    violations.append(f"{path.relative_to(repo)}:{node.lineno}")
    assert violations == []
