"""Client-separated stems and forked jobs (Dub Anything).

A client that separates the voices itself uploads with ``separate`` false and
attaches its stems; one that renders the same audio in a second voice forks
the job instead of uploading it again. Both are about the job record: renders,
transcription and exports read the stems from ``vocals_path`` and
``no_vocals_path``, so those must name the client's files.
"""

import os

import numpy as np
import pytest
import soundfile as sf

from services import dub_pipeline


@pytest.fixture
def jobs(tmp_path, monkeypatch):
    from core.db import init_db

    init_db()  # an unknown job is looked up in the (test) database
    root = tmp_path / "dub_jobs"
    root.mkdir()
    monkeypatch.setattr(dub_pipeline, "DUB_DIR", str(root))
    monkeypatch.setattr(dub_pipeline, "_DUB_DIR_REAL", os.path.realpath(root))
    saved = {}
    monkeypatch.setattr(dub_pipeline, "_persist_job", lambda job_id, job, *a, **k: saved.__setitem__(job_id, dict(job)))
    dub_pipeline._dub_jobs.clear()
    yield root, saved
    dub_pipeline._dub_jobs.clear()


def wav(path, seconds, sr=16000):
    sf.write(str(path), np.zeros(int(seconds * sr), dtype="float32"), sr)
    return str(path)


def ingested(root, job_id, seconds=3.0):
    """A job as an upload with separate=false leaves it: the mix stands in for the voices."""
    folder = root / job_id
    folder.mkdir()
    (folder / "original.mp4").write_bytes(b"video")
    audio = wav(folder / "audio.wav", seconds)
    wav(folder / "audio_hq.wav", seconds, sr=44100)
    dub_pipeline.put_job(
        job_id,
        {
            "video_path": str(folder / "original.mp4"),
            "audio_path": audio,
            "vocals_path": audio,
            "no_vocals_path": None,
            "thumb_path": None,
            "duration": seconds,
            "filename": "part.mp4",
            "segments": None,
            "dubbed_tracks": {},
            "scene_cuts": [],
            "input_type": "video",
        },
    )
    return folder


def test_attached_stems_become_the_jobs_voices_and_bed(jobs):
    root, saved = jobs
    folder = ingested(root, "part-01")
    wav(folder / "vocals.wav", 3.0, sr=44100)
    wav(folder / "no_vocals.wav", 3.0, sr=44100)

    updates = dub_pipeline.attach_stems("part-01", "vocals.wav", "no_vocals.wav")

    job = dub_pipeline.get_job("part-01")
    assert job["vocals_path"] == os.path.realpath(folder / "vocals.wav") == updates["vocals_path"]
    assert job["no_vocals_path"] == os.path.realpath(folder / "no_vocals.wav")
    assert saved["part-01"]["vocals_path"] == job["vocals_path"], "persisted, not only in memory"


@pytest.mark.parametrize("name", ["../vocals.wav", "sub/vocals.wav", "missing.wav", "", ".."])
def test_stems_must_be_files_in_the_job_folder(jobs, name):
    root, _ = jobs
    ingested(root, "part-01")
    wav(root / "vocals.wav", 3.0)  # outside the job folder
    with pytest.raises(ValueError):
        dub_pipeline.attach_stems("part-01", name, None)


def test_a_stem_of_another_length_is_refused(jobs):
    """Stems cut from another part would misplace every line."""
    root, _ = jobs
    folder = ingested(root, "part-01", seconds=3.0)
    wav(folder / "vocals.wav", 9.0)
    with pytest.raises(ValueError, match="lasts"):
        dub_pipeline.attach_stems("part-01", "vocals.wav", None)
    assert dub_pipeline.get_job("part-01")["vocals_path"].endswith("audio.wav")


def test_unreadable_stems_are_refused(jobs):
    root, _ = jobs
    folder = ingested(root, "part-01")
    (folder / "vocals.wav").write_bytes(b"not audio")
    with pytest.raises(ValueError, match="readable"):
        dub_pipeline.attach_stems("part-01", "vocals.wav", None)


def test_attaching_to_an_unknown_job_is_a_lookup_error(jobs):
    with pytest.raises(LookupError):
        dub_pipeline.attach_stems("nobody", "vocals.wav", None)


def test_a_fork_shares_the_media_and_none_of_the_render_state(jobs):
    root, saved = jobs
    folder = ingested(root, "part-01")
    wav(folder / "vocals.wav", 3.0, sr=44100)
    wav(folder / "no_vocals.wav", 3.0, sr=44100)
    dub_pipeline.attach_stems("part-01", "vocals.wav", "no_vocals.wav")
    job = dub_pipeline.get_job("part-01")
    job.update(segments=[{"id": "s1"}], speaker_clones={"A": "x"}, seg_hashes={"s1": "h"}, dubbed_tracks={"en": {}})
    (folder / "seg_en_L1.wav").write_bytes(b"take")

    dub_pipeline.fork_job("part-01", "part-01-us")

    fork = dub_pipeline.get_job("part-01-us")
    new = root / "part-01-us"
    for key, name in (("vocals_path", "vocals.wav"), ("no_vocals_path", "no_vocals.wav"), ("audio_path", "audio.wav")):
        assert fork[key] == str(new / name)
        assert os.path.samefile(fork[key], folder / name), f"{name} is shared, not separated again"
    assert os.path.samefile(new / "audio_hq.wav", folder / "audio_hq.wav")
    assert fork["video_path"] == str(new / "original.mp4")
    assert fork["segments"] is None and fork["dubbed_tracks"] == {} and fork["forked_from"] == "part-01"
    assert "speaker_clones" not in fork and "seg_hashes" not in fork
    assert not (new / "seg_en_L1.wav").exists(), "takes stay with the job that made them"
    assert saved["part-01-us"]["vocals_path"] == fork["vocals_path"]
    assert dub_pipeline.get_job("part-01")["speaker_clones"] == {"A": "x"}, "the source is untouched"


def test_a_fork_keeps_paths_outside_the_source_folder(jobs):
    root, _ = jobs
    ingested(root, "part-01")
    elsewhere = wav(root / "shared-vocals.wav", 3.0)
    dub_pipeline.get_job("part-01")["vocals_path"] = elsewhere
    dub_pipeline.fork_job("part-01", "part-01-us")
    assert dub_pipeline.get_job("part-01-us")["vocals_path"] == elsewhere


def test_a_taken_id_is_refused_and_left_alone(jobs):
    root, _ = jobs
    ingested(root, "part-01")
    (root / "part-01-us").mkdir()
    (root / "part-01-us" / "keep.txt").write_text("mine")
    with pytest.raises(FileExistsError):
        dub_pipeline.fork_job("part-01", "part-01-us")
    assert (root / "part-01-us" / "keep.txt").read_text() == "mine"


@pytest.mark.parametrize("new_id", ["../escape", "a/b", "", "has space"])
def test_unsafe_fork_ids_are_refused(jobs, new_id):
    root, _ = jobs
    ingested(root, "part-01")
    with pytest.raises(ValueError):
        dub_pipeline.fork_job("part-01", new_id)


def test_forking_an_unknown_job_creates_nothing(jobs):
    root, _ = jobs
    with pytest.raises(LookupError):
        dub_pipeline.fork_job("nobody", "nobody-us")
    assert not (root / "nobody-us").exists()


def test_a_job_still_ingesting_can_be_neither_forked_nor_given_stems(jobs):
    root, _ = jobs
    folder = ingested(root, "part-01")
    wav(folder / "vocals.wav", 3.0)
    dub_pipeline.begin_ingest("part-01")
    try:
        with pytest.raises(ValueError, match="prepared"):
            dub_pipeline.fork_job("part-01", "part-01-us")
        with pytest.raises(ValueError, match="prepared"):
            dub_pipeline.attach_stems("part-01", "vocals.wav", None)
    finally:
        dub_pipeline.end_ingest("part-01")
    assert not (root / "part-01-us").exists()


def test_an_unseparated_upload_neither_offers_nor_reuses_cached_stems():
    """Its stems do not exist until the client attaches them, so it must not
    hand the mix to a later upload as "vocals", nor take another job's."""
    import inspect

    src = inspect.getsource(dub_pipeline.ingest_pipeline)
    assert 'separate = source.get("separate", True)' in src
    assert "compute_file_hash, audio_path) if separate else \"\"" in src
