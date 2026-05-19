# Reproducible LaTeX build environment for paper/main.tex
# Pinned to TUG texlive 2024 on Debian Bookworm (slim).
# Usage:
#   docker build -t tessera-paper .
#   docker run --rm -v $(pwd):/paper tessera-paper
FROM texlive/texlive:TL2024-historic-slim

WORKDIR /paper
COPY . .

RUN pdflatex -interaction=nonstopmode main.tex && \
    bibtex main && \
    pdflatex -interaction=nonstopmode main.tex && \
    pdflatex -interaction=nonstopmode main.tex

# Output: /paper/main.pdf
