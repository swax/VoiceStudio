# VoiceStudio for Dub Anything

This fork integrates upstream commit
`2e195d72d3794a88b7cb37f87a3e92a20193574e`. Its original baseline was
`c52618afa75f5a838228fe4eeac6301e790bafcf`, with backend changes shipped as
`voicestudio-local-fixes.patch`; those moved here on 2026-11-07.

- Respect the source-language hint during dubbing transcription.
- Add per-segment voice and emotion references and IndexTTS seeds, including
  these optional inputs in take fingerprints while preserving old defaults.
- Add echo and public-address effects with reverb tails.
- Adapt the backend and IndexTTS sidecar to the legacy `2.9.1+rocm7.13.x` Torch
  build on Windows; leave other runtimes' upstream dispatch policy intact.
- Preserve LF bytes for hash-pinned bundled model text assets on Windows.

CPU float32 and IndexTTS speed are now provided by upstream. The old early-cache
marker was retired in favor of upstream's atomic publication rules. Reusing
failed renders' takes remains a separate follow-up that must keep unaccepted
audio out of published previews.

Dub Anything pins a tested commit of this repository as `vendor/VoiceStudio`.
Keep upstream updates deliberate: merge them here, verify the dubbing flow,
then update Dub Anything's submodule pointer. Setup uses the recorded commit,
without following the latest branch head.

The upstream license and notices remain in `LICENSE` and `LICENSE-NOTICE.md`.
Model weights retain their own licenses. Media, voice references, generated
audio, model downloads and local environments are not part of this fork.
