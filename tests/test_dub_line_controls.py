"""Fork controls preserve backend compatibility and invalidate conditioned takes."""

import ast
import importlib.util
import os
from pathlib import Path
import sys
import tempfile
from types import SimpleNamespace

import pytest

from schemas.requests import DubSegment
from services.incremental import segment_fingerprint

ROOT = Path(__file__).resolve().parents[1]


@pytest.mark.parametrize(
    "key,value",
    [
        ("ref_audio", "/reference.wav"),
        ("ref_text", "Reference words."),
        ("emo_audio", "/emotion.wav"),
        ("emo_alpha", 0.5),
        ("seed", 0),
    ],
)
def test_conditioning_changes_invalidate_a_take(key, value):
    segment = {"start": 0, "end": 1, "text": "Hello there."}
    assert segment_fingerprint(segment) != segment_fingerprint({**segment, key: value})


def test_unset_controls_keep_raw_and_parsed_fingerprints_identical():
    segment = {"start": 0, "end": 1, "text": "Hello there."}
    parsed = DubSegment(**segment).model_dump()
    assert segment_fingerprint(segment) == segment_fingerprint(parsed)


@pytest.mark.parametrize("hint,supports_language", [(None, True), ("ja", True), ("ja", False)])
def test_transcription_worker_pins_language_only_for_supported_backends(hint, supports_language):
    # Exercise the real nested chunk worker without loading ASR or a GPU.
    source = (ROOT / "backend/api/routers/dub_core.py").read_text(encoding="utf-8")
    worker = next(
        n for n in ast.walk(ast.parse(source)) if isinstance(n, ast.FunctionDef) and n.name == "_transcribe_chunk"
    )
    calls = []

    def with_language(path, word_timestamps=True, language=None):
        calls.append(language)
        return {"chunks": [], "language": language}

    def without_language(path, word_timestamps=True):
        calls.append(None)
        return {"chunks": [], "language": "en"}

    scope = {
        "chunk_arr": [],
        "t0": 0,
        "sr": 16000,
        "job": {"source_lang_override": hint},
        "_asr_backend": SimpleNamespace(transcribe=with_language if supports_language else without_language),
        "_safe_soundfile_write": lambda *args: None,
        "tempfile": tempfile,
        "os": os,
    }
    exec(
        compile(ast.Module(body=[worker], type_ignores=[]), str(ROOT / "backend/api/routers/dub_core.py"), "exec"),
        scope,
    )
    scope["_transcribe_chunk"]()
    assert calls == [hint if supports_language else None]


@pytest.mark.parametrize(
    "version,legacy",
    [
        ("2.9.1+rocm7.13.0", True),
        ("2.9.1+rocm7.2.1", False),
        ("2.11.0+rocm7.2", False),
    ],
)
def test_legacy_rocm_override_does_not_change_modern_runtime(monkeypatch, version, legacy):
    from services.model_manager import _configure_rocm_windows

    torch = SimpleNamespace(
        __version__=version,
        version=SimpleNamespace(hip="7.2"),
        backends=SimpleNamespace(cudnn=SimpleNamespace(enabled=True)),
    )
    monkeypatch.setattr(sys, "platform", "win32")
    monkeypatch.delenv("OMNIVOICE_MIOPEN", raising=False)
    monkeypatch.delenv("TORCH_ROCM_AOTRITON_ENABLE_EXPERIMENTAL", raising=False)
    _configure_rocm_windows(torch)
    assert torch.backends.cudnn.enabled is not legacy
    assert ("TORCH_ROCM_AOTRITON_ENABLE_EXPERIMENTAL" in os.environ) is legacy


@pytest.mark.parametrize("version", ["2.9.1+rocm7.2.1", "2.11.0+rocm7.2"])
def test_sidecar_keeps_modern_rocm_dispatch_and_precision(monkeypatch, version):
    import runpy

    sidecar = runpy.run_path(str(ROOT / "backend/engines/indextts/main.py"))
    torch = SimpleNamespace(
        __version__=version,
        version=SimpleNamespace(hip="7.2"),
        backends=SimpleNamespace(cudnn=SimpleNamespace(enabled=True)),
        float16="float16",
        bfloat16="bfloat16",
    )
    monkeypatch.setitem(sys.modules, "torch", torch)
    monkeypatch.setattr(importlib.util, "find_spec", lambda name: object())
    monkeypatch.setattr(sys, "platform", "win32")
    sidecar["_configure_rocm_windows"]()
    inference = object()
    model = SimpleNamespace(
        dtype="bfloat16", s2mel=SimpleNamespace(models={"cfm": SimpleNamespace(inference=inference)})
    )
    sidecar["_rocm_reduced_precision_s2mel"](model)
    assert torch.backends.cudnn.enabled is True
    assert model.s2mel.models["cfm"].inference is inference
