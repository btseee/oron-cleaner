# Progress: clip recovery

Plan: docs/superpowers/plans/2026-09-05-clip-recovery.md
Branch: feat/clip-recovery
Retargeted at WorldSpeech; Common Voice fires none of these gates (see spec).

Task 1: complete (commits 49f8c3f..684a99c, review clean after one fix round)
  Fixed: my brief mapped Cyrillic-range i U+0456 as a "Latin" homoglyph and to the wrong letter; space-only splitting rewrote the Latin half of "Google-ийн"; NaN/inf made a repair return an entirely-NaN clip as a success.
  Minor, for final triage:
  - test_an_infinite_sample_is_refused_not_amplified passes with or without the new guard (peak=inf already trips RECOVERY_QUIET_PEAK); its docstring overclaims
  - mypy is in CONTRIBUTING but not in CI, so the _REPAIRS signature pin only fires locally
Also fixed 3 ruff errors that predated this branch on main (9227118) -- my own thread-clamp commit broke oron-cleaner's lint gate this morning.
