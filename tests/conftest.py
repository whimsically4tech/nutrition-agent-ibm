"""
tests/conftest.py
─────────────────────────────────────────────────────────────
Test configuration and global fixtures.
Pre-stubs the IBM watsonx.ai SDK so all agent modules can be
imported without real credentials in the test environment.
"""
import os
import sys
from types import ModuleType
from unittest.mock import MagicMock

# ── Set required env vars before any Settings import ─────────
os.environ.setdefault("WATSONX_APIKEY", "test-api-key")
os.environ.setdefault("WATSONX_URL", "https://us-south.ml.cloud.ibm.com")
os.environ.setdefault("WATSONX_PROJECT_ID", "test-project-id")


def _make_stub_module(name: str) -> ModuleType:
    """Create a stub module with any attribute access returning a MagicMock."""
    mod = ModuleType(name)
    mod.__spec__ = None  # type: ignore[assignment]
    return mod


def _stub_ibm_watsonx_ai() -> None:
    """
    Inject stub modules for the entire ibm_watsonx_ai namespace into sys.modules
    before any src.* imports can trigger the real import.
    """
    if "ibm_watsonx_ai" in sys.modules:
        return  # already installed or already stubbed

    # Root package
    root = _make_stub_module("ibm_watsonx_ai")
    root.APIClient = MagicMock()
    root.Credentials = MagicMock()

    # Sub-packages
    fm = _make_stub_module("ibm_watsonx_ai.foundation_models")
    fm.ModelInference = MagicMock()

    embeddings = _make_stub_module("ibm_watsonx_ai.foundation_models.embeddings")
    embeddings.Embeddings = MagicMock()

    metanames = _make_stub_module("ibm_watsonx_ai.metanames")
    # GenTextParamsMetaNames needs to behave like a namespace of string constants
    GenParams = MagicMock()
    GenParams.DECODING_METHOD = "decoding_method"
    GenParams.MAX_NEW_TOKENS = "max_new_tokens"
    GenParams.MIN_NEW_TOKENS = "min_new_tokens"
    GenParams.TEMPERATURE = "temperature"
    GenParams.REPETITION_PENALTY = "repetition_penalty"
    metanames.GenTextParamsMetaNames = GenParams

    sys.modules["ibm_watsonx_ai"] = root
    sys.modules["ibm_watsonx_ai.foundation_models"] = fm
    sys.modules["ibm_watsonx_ai.foundation_models.embeddings"] = embeddings
    sys.modules["ibm_watsonx_ai.metanames"] = metanames


# ── Stub IBM SDK before any src imports ──────────────────────
_stub_ibm_watsonx_ai()

# ── Pre-import agent submodules so patch() can resolve them ──
# This ensures sys.modules["src.agents.knowledge_agent"] etc. exist
# before unittest.mock.patch tries to resolve the dotted path.
import src.agents.guardrails          # noqa: E402, F401
import src.agents.knowledge_agent     # noqa: E402, F401
import src.agents.recommendation_agent  # noqa: E402, F401
import src.agents.health_agent        # noqa: E402, F401
import src.agents.food_log_agent      # noqa: E402, F401
import src.rag.ingest                 # noqa: E402, F401
import src.rag.retriever              # noqa: E402, F401
