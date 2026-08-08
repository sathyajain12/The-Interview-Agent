import json
import os
import sys
from pathlib import Path

# Tests must exercise the deterministic engine, never spend tokens.
os.environ.setdefault("INTERVIEW_FORCE_OFFLINE", "1")

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import pytest  # noqa: E402

from app.main import app  # noqa: E402


@pytest.fixture(scope="session")
def candidates() -> list[dict]:
    raw = json.loads((ROOT / "data" / "candidates.json").read_text(encoding="utf-8"))
    return raw["candidates"]


@pytest.fixture()
def client():
    from fastapi.testclient import TestClient

    with TestClient(app) as c:
        yield c
