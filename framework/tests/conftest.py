"""Shared pytest fixtures for framework tests."""
from pathlib import Path
import pytest

REPO_ROOT = Path(__file__).resolve().parents[2]
FIXTURES_DIR = Path(__file__).parent / "fixtures"


@pytest.fixture
def repo_root() -> Path:
    return REPO_ROOT


@pytest.fixture
def fixtures_dir() -> Path:
    return FIXTURES_DIR


@pytest.fixture
def fake_arena_dir() -> Path:
    return FIXTURES_DIR / "fake_arena"


@pytest.fixture
def fake_split_arena_dir() -> Path:
    return FIXTURES_DIR / "fake_split_arena"


@pytest.fixture
def fake_registry_path() -> Path:
    return FIXTURES_DIR / "fake_registry.yaml"
