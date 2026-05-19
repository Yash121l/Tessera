# Tessera Research Paper

IEEE two-column format, ~12 pages. Covers triple-barrier labeling, CPCV,
deflated Sharpe validation, model comparison, ablations, and pitfalls.

## Compiled PDF

[paper/main.pdf](main.pdf) — committed to this repo.

## Build from source

**Local (requires tectonic):**
```bash
# Install tectonic (auto-fetches packages — no full TeX install needed)
brew install tectonic         # macOS
cargo install tectonic        # Linux

# From this directory:
tectonic main.tex
# Output: main.pdf
```

**Docker (fully reproducible, pinned texlive 2024):**
```bash
docker build -t tessera-paper .
docker run --rm -v $(pwd):/paper tessera-paper
# Output: main.pdf
```

**Makefile shortcut (from repo root):**
```bash
make compile-paper
```

## Files

| File | Purpose |
|---|---|
| `main.tex` | Full LaTeX source |
| `refs.bib` | BibTeX references |
| `main.pdf` | Compiled output (committed) |
| `figures/` | All figures referenced by the paper |
| `demo_script.md` | Voiceover script for screen-recording demos |
| `demo/` | Demo artifacts (cast.cast, record_demo.sh) |
| `Dockerfile` | Pinned texlive 2024 build environment |

## Bumping the pin

To update to a newer texlive version, change `TL2024-historic-slim` in
`Dockerfile` to the desired tag (see DockerHub texlive/texlive).
Recompile and commit the new `main.pdf`.
