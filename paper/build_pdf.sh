#!/usr/bin/env bash
# Build paper/draft.pdf from paper/draft.md (YAML front matter + custom typst layout).
set -euo pipefail
cd "$(dirname "$0")"

pandoc draft.md \
  -o draft.pdf \
  --pdf-engine=typst \
  --template=pandoc.typ \
  --resource-path=. \
  -H pdf_header.typ \
  -V fontsize=10pt \
  -V papersize=us-letter \
  -V margin-top=1in \
  -V margin-bottom=1in \
  -V margin-left=1.05in \
  -V margin-right=1.05in

echo "Wrote $(pwd)/draft.pdf"
