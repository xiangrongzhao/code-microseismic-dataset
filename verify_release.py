"""Check the integrity of the downloaded repository files against SHA-256."""
import hashlib
from pathlib import Path

ROOT = Path(__file__).resolve().parent


def digest(path):
    result = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            result.update(chunk)
    return result.hexdigest()


def main():
    errors = []
    lines = (ROOT / "SHA256SUMS").read_text(encoding="utf-8").splitlines()
    for line in lines:
        expected, relative = line.split("  ", 1)
        path = ROOT / relative
        if not path.is_file() or digest(path) != expected:
            errors.append(relative)
    if errors:
        raise SystemExit("Missing or changed:\n" + "\n".join(errors))
    print(f"Verified {len(lines)} files.")


if __name__ == "__main__":
    main()
