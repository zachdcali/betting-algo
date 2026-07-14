"""Validate and stage the checksum-pinned model release archive.

The hourly runner downloads one repository release asset on a model-cache miss.
Never extract that tarball directly into the checkout: even a later per-model
checksum check cannot undo a path traversal or an overwritten Python module.
This helper pins the complete archive, restricts every member to the model
results tree, rejects links/devices/duplicates, extracts into a temporary
directory, and only then copies collision-free files into the checkout.
"""
from __future__ import annotations

import argparse
from hashlib import sha256
from pathlib import Path, PurePosixPath
import shutil
import tarfile
import tempfile


ALLOWED_PREFIX = PurePosixPath("results/professional_tennis")
COPY_CHUNK_BYTES = 1024 * 1024
MAX_ARCHIVE_MEMBERS = 10_000
MAX_UNCOMPRESSED_BYTES = 2 * 1024 * 1024 * 1024


class UnsafeModelArchive(RuntimeError):
    """Raised when a release archive violates the extraction contract."""


def file_sha256(path: Path) -> str:
    digest = sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(COPY_CHUNK_BYTES):
            digest.update(chunk)
    return digest.hexdigest()


def _safe_relative_path(name: str) -> PurePosixPath:
    if not name or "\\" in name:
        raise UnsafeModelArchive(f"invalid archive member path: {name!r}")
    path = PurePosixPath(name)
    if path.is_absolute() or any(part in {"", ".", ".."} for part in path.parts):
        raise UnsafeModelArchive(f"unsafe archive member path: {name!r}")
    try:
        path.relative_to(ALLOWED_PREFIX)
    except ValueError as exc:
        raise UnsafeModelArchive(
            f"archive member is outside {ALLOWED_PREFIX.as_posix()!r}: {name!r}"
        ) from exc
    return path


def validate_members(archive: tarfile.TarFile) -> tuple[tarfile.TarInfo, ...]:
    members = tuple(archive.getmembers())
    if not members:
        raise UnsafeModelArchive("model release archive is empty")
    if len(members) > MAX_ARCHIVE_MEMBERS:
        raise UnsafeModelArchive(
            f"model release archive has too many members: {len(members)}"
        )
    uncompressed_bytes = sum(member.size for member in members if member.isfile())
    if uncompressed_bytes > MAX_UNCOMPRESSED_BYTES:
        raise UnsafeModelArchive(
            "model release archive exceeds the uncompressed size limit: "
            f"{uncompressed_bytes} bytes"
        )
    seen: set[str] = set()
    file_count = 0
    for member in members:
        path = _safe_relative_path(member.name)
        normalized = path.as_posix()
        if normalized in seen:
            raise UnsafeModelArchive(f"duplicate archive member: {normalized!r}")
        seen.add(normalized)
        if member.isfile():
            file_count += 1
        elif not member.isdir():
            raise UnsafeModelArchive(
                f"archive member is not a regular file or directory: {normalized!r}"
            )
    if not file_count:
        raise UnsafeModelArchive("model release archive contains no files")
    return members


def extract_model_release(
    archive_path: str | Path,
    destination_root: str | Path,
    expected_sha256: str,
) -> int:
    archive_path = Path(archive_path).resolve()
    destination_root = Path(destination_root).resolve()
    expected = expected_sha256.strip().lower()
    if len(expected) != 64 or any(character not in "0123456789abcdef" for character in expected):
        raise ValueError("expected SHA-256 must be 64 lowercase hexadecimal characters")
    actual = file_sha256(archive_path)
    if actual != expected:
        raise UnsafeModelArchive(
            f"model release SHA-256 mismatch: expected {expected}, got {actual}"
        )

    destination_root.mkdir(parents=True, exist_ok=True)
    with tarfile.open(archive_path, mode="r:gz") as archive:
        members = validate_members(archive)
        with tempfile.TemporaryDirectory(
            prefix=".model-release-", dir=destination_root
        ) as temporary:
            staging = Path(temporary)
            archive.extractall(staging, members=members, filter="data")
            extracted_root = staging / Path(ALLOWED_PREFIX.as_posix())
            target_root = destination_root / Path(ALLOWED_PREFIX.as_posix())
            collisions = sorted(
                path.relative_to(extracted_root).as_posix()
                for path in extracted_root.rglob("*")
                if path.is_file()
                and (target_root / path.relative_to(extracted_root)).exists()
            )
            if collisions:
                raise UnsafeModelArchive(
                    "model release would overwrite existing files: "
                    + ", ".join(collisions[:5])
                )
            shutil.copytree(extracted_root, target_root, dirs_exist_ok=True)
    return sum(member.isfile() for member in members)


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--archive", required=True)
    parser.add_argument("--destination-root", default=".")
    parser.add_argument("--expected-sha256", required=True)
    return parser


def main() -> int:
    args = _parser().parse_args()
    count = extract_model_release(
        args.archive, args.destination_root, args.expected_sha256
    )
    print(f"validated and staged {count} model release files")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
