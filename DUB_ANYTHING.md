# VoiceStudio for Dub Anything

This fork carries Dub Anything's backend changes on upstream commit
`c52618afa75f5a838228fe4eeac6301e790bafcf`. The changes were previously shipped
as `voicestudio-local-fixes.patch`; they moved here on 2026-11-07.

- Respect the source-language hint during dubbing transcription.
- Use float32 for OmniVoice on the CPU.
- Keep generated takes reusable after a timing overflow.
- Add per-segment voice and emotion references and IndexTTS seeds and speed.
- Add echo and public-address effects with reverb tails.
- Adapt the backend and IndexTTS sidecar to ROCm on Windows.

Dub Anything pins a tested commit of this repository as `vendor/VoiceStudio`.
Keep upstream updates deliberate: merge them here, verify the dubbing flow,
then update Dub Anything's submodule pointer. Setup uses the recorded commit,
without following the latest branch head.

The upstream license and notices remain in `LICENSE` and `LICENSE-NOTICE.md`.
Model weights retain their own licenses. Media, voice references, generated
audio, model downloads and local environments are not part of this fork.
