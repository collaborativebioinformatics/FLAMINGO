# FLAMINGO presentation

Four visual slides: project aim, methods, saved simulation results, and the dashboard.

- `FLAMINGO.pptx`: PowerPoint, with explanation and source references in speaker notes. Slide artwork is embedded at high resolution.
- `FLAMINGO.pdf`: vector PDF for presenting or sharing.
- `index.html`: local browser presentation; arrow keys navigate, F enters fullscreen.
- `preview.png`: four-slide overview.
- `speaker-notes.md`: plain-text narration and provenance.
- `assets/slide-*.svg`: scalable slide artwork.

The results chart uses the committed quadratic FedMR CSV and truth JSON; no new experiment is needed. The dashboard image is an actual local screenshot. Keeping records local does not itself imply a formal privacy guarantee.

To rebuild, install `matplotlib`, `numpy`, `Pillow`, and `python-pptx`, then run `python presentation/build.py` from the repository root. The builder reuses `assets/dashboard.png`.
