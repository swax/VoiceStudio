"""POST /dub/stems/{job_id} and POST /dub/fork/{job_id}, and the upload's separate flag."""
from __future__ import annotations

import os
import uuid

import numpy as np
import pytest
import soundfile as sf

os.environ.setdefault("OMNIVOICE_MODEL", "test")


@pytest.fixture()
def client():
    from fastapi.testclient import TestClient
    from core.db import init_db
    from main import app

    init_db()
    return TestClient(app, client=("127.0.0.1", 50000))


@pytest.fixture()
def job():
    """A prepared job whose voices are not separated yet, in the real job folder."""
    import shutil

    from services import dub_pipeline

    jid = "st" + uuid.uuid4().hex[:8]
    folder = dub_pipeline.safe_job_dir(jid)
    os.makedirs(folder)
    audio = os.path.join(folder, "audio.wav")
    sf.write(audio, np.zeros(16000 * 2, dtype="float32"), 16000)
    sf.write(os.path.join(folder, "vocals.wav"), np.zeros(44100 * 2, dtype="float32"), 44100)
    dub_pipeline.put_job(jid, {"audio_path": audio, "vocals_path": audio, "no_vocals_path": None,
                               "duration": 2.0, "filename": "part.mp4", "segments": None,
                               "dubbed_tracks": {}, "scene_cuts": []})
    yield jid
    for j in (jid, jid + "-us"):
        dub_pipeline._dub_jobs.pop(j, None)
        shutil.rmtree(dub_pipeline.safe_job_dir(j), ignore_errors=True)


def test_stems_are_attached(client, job):
    res = client.post(f"/dub/stems/{job}", json={"vocals": "vocals.wav"})
    assert res.status_code == 200, res.text
    assert res.json()["vocals_path"].endswith("vocals.wav")
    assert res.json()["no_vocals_path"] is None


def test_a_bad_stem_is_a_400_and_an_unknown_job_a_404(client, job):
    assert client.post(f"/dub/stems/{job}", json={"vocals": "../x.wav"}).status_code == 400
    assert client.post("/dub/stems/nojob123", json={"vocals": "vocals.wav"}).status_code == 404


def test_a_fork_is_made_once(client, job):
    res = client.post(f"/dub/fork/{job}", json={"job_id": job + "-us"})
    assert res.status_code == 200, res.text
    assert res.json() == {"job_id": job + "-us", "forked_from": job, "duration": 2.0}
    assert client.post(f"/dub/fork/{job}", json={"job_id": job + "-us"}).status_code == 409


def test_fork_ids_are_validated(client, job):
    assert client.post(f"/dub/fork/{job}", json={"job_id": "../up"}).status_code == 422
    assert client.post("/dub/fork/nojob123", json={"job_id": "nojob123-us"}).status_code == 404


def test_the_upload_passes_separate_to_the_ingest(monkeypatch, client):
    from api.routers import dub_core

    seen = {}

    async def add_task(task_id, kind, gen, job_id, job_dir, source, filename):
        seen.update(source)

    monkeypatch.setattr(dub_core.task_manager, "add_task", add_task)
    res = client.post("/dub/upload", files={"video": ("part.wav", b"RIFF0000WAVE", "audio/wav")},
                      data={"input_type": "audio", "separate": "false"})
    assert res.status_code == 202, res.text
    assert seen["separate"] is False
    import shutil
    from services import dub_pipeline
    shutil.rmtree(dub_pipeline.safe_job_dir(res.json()["job_id"]), ignore_errors=True)
