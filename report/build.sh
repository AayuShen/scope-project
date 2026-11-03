#!/usr/bin/env bash
# Rebuild the report: numbers, tables and figures from the result files, the bibliography, then the PDF.
#   bash report/build.sh        (Git Bash on Windows; needs WSL's scope env for matplotlib and MiKTeX's pdflatex)
set -euo pipefail
cd "$(dirname "$0")"
PDFLATEX="${PDFLATEX:-/e/DevTools/MiKTeX/texmf/miktex/bin/x64/pdflatex.exe}"
MSYS_NO_PATHCONV=1 wsl.exe -d Ubuntu-26.04 -u aayush -- /home/aayush/miniforge3/envs/scope/bin/python make_assets.py
node make_bib.js
for _ in 1 2; do "$PDFLATEX" -interaction=nonstopmode -halt-on-error main.tex > /dev/null; done
grep -E "^!|undefined" main.log || true
grep -o "Output written on .*" main.log
