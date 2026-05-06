"""Upload the final ParrotLabs submission checkpoint to a public Hugging Face repo.

Idempotent: re-running is safe. Creates the model repo if missing
(``exist_ok=True``), then uploads ``parrotlabs_final.pt`` to the root of
``main`` and verifies remote presence and SHA-256.

Run:
    uv run python tools/upload_submission_to_hf.py

Requires ``HF_TOKEN`` (loaded from ``.env`` via ``python-dotenv`` if available,
otherwise read from the process environment).
"""

from __future__ import annotations

import hashlib
import os
import sys
from pathlib import Path

from huggingface_hub import HfApi, create_repo
from huggingface_hub.utils import HfHubHTTPError

try:
    from dotenv import load_dotenv

    load_dotenv()
except ImportError:  # pragma: no cover - optional convenience
    pass


REPO_ID = "ParrotLabs/parrotlabs_parrotllm"
REPO_TYPE = "model"
PROJECT_ROOT = Path(__file__).resolve().parents[1]
LOCAL_PT = PROJECT_ROOT / "Submissions" / "parrotlabs_parrotllm" / "runs" / "parrotlabs_final.pt"
TARGET = "parrotlabs_final.pt"
EXPECTED_SHA = "1c131cd13b088e875e0705f5a428fffac394005d8f61c947421c2be8c87bf888"


def _sha256(path: Path) -> str:
    """Stream-hash so we don't load 458 MB into memory."""
    h = hashlib.sha256()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def main() -> int:
    token = os.environ.get("HF_TOKEN")
    if not token:
        print("ERROR: HF_TOKEN not set (export it or put it in .env)", file=sys.stderr)
        return 2
    if not LOCAL_PT.exists():
        print(f"ERROR: checkpoint not found at {LOCAL_PT}", file=sys.stderr)
        return 2

    api = HfApi(token=token)

    # 1) Create repo (public). Idempotent via exist_ok=True.
    try:
        url = create_repo(
            REPO_ID,
            repo_type=REPO_TYPE,
            private=False,
            token=token,
            exist_ok=True,
        )
        print(f"Repo ready: {url}")
    except HfHubHTTPError as exc:  # pragma: no cover - surfaced to caller
        print(f"ERROR: create_repo failed: {exc}", file=sys.stderr)
        return 1

    # 2) Local SHA for verification.
    local_sha = _sha256(LOCAL_PT)
    print(f"Local SHA-256: {local_sha}")
    if local_sha != EXPECTED_SHA:
        print(
            f"WARNING: local SHA-256 differs from MANIFEST expectation\n"
            f"  expected: {EXPECTED_SHA}\n"
            f"  actual:   {local_sha}",
            file=sys.stderr,
        )

    # 3) Upload (idempotent: re-uploading the same bytes is a no-op commit).
    short_sha = local_sha[:12]
    commit_message = (
        f"Upload {TARGET} (SHA-256 {short_sha}...) - public bench avg 38.38%"
    )
    upload_url = api.upload_file(
        path_or_fileobj=str(LOCAL_PT),
        path_in_repo=TARGET,
        repo_id=REPO_ID,
        repo_type=REPO_TYPE,
        commit_message=commit_message,
    )
    print(f"Uploaded: {upload_url}")

    # 4) Verify presence on main.
    files = api.list_repo_files(REPO_ID, repo_type=REPO_TYPE)
    if TARGET not in files:
        print(
            f"ERROR: {TARGET} not present in remote files after upload: {files}",
            file=sys.stderr,
        )
        return 1
    print("Verified remote presence on main.")

    # 5) Cross-check the remote LFS sha256 if available.
    info = api.repo_info(REPO_ID, repo_type=REPO_TYPE, files_metadata=True)
    remote_sha = None
    for sib in getattr(info, "siblings", []) or []:
        if sib.rfilename == TARGET:
            remote_sha = getattr(sib, "lfs", None)
            if isinstance(remote_sha, dict):
                remote_sha = remote_sha.get("sha256")
            break
    if remote_sha and remote_sha != local_sha:
        print(
            f"WARNING: remote SHA mismatch\n"
            f"  local:  {local_sha}\n"
            f"  remote: {remote_sha}",
            file=sys.stderr,
        )
    elif remote_sha:
        print(f"Remote SHA-256 matches local: {remote_sha}")

    print(f"Public model page: https://huggingface.co/{REPO_ID}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
