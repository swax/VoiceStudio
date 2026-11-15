# VoiceStudio for Dub Anything

Fork-specific behavior beyond upstream `9ce1333fe55c1bf9de6992e762679aea23f6c5b1`:

- **Per-line direction:** accept voice references and transcripts, emotion references
  and strengths, and seed overrides for individual dub segments. Include these
  controls in incremental fingerprints and private crash-recovery identities.
- **Audio effects:** add echo and public-address presets with reverb tails.
- **Shared media:** let uploads skip built-in separation, attach client-separated
  stems, and fork jobs over hard-linked media for another voice. Stem attachment
  and job forking require administrator access.
- **Responsive watermarking:** mark fresh dub segments on the existing watermark
  worker; recovered segments keep their existing mark.
- **Legacy Windows AMD support:** retain compatibility adjustments for the
  `2.9.1+rocm7.13.x` Torch build in the backend and IndexTTS sidecar.
- **Windows checkout:** preserve LF bytes in hash-pinned bundled model text assets
  and generated mobile contracts.
