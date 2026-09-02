#!/usr/bin/env python3
"""Build a clean Word manuscript from new.tex.

4.docx started life as a PDF-to-Word conversion, and it showed: two of its
tables survived only as loose fragments scattered through the body text, the
classifier list was interleaved with the index equations, and several sentences
were cut in half by a floating text box. Patching that in place was possible
and was in fact done, but it is repair work on a derived artefact, and it left
two files that could drift apart.

Converting from the LaTeX source removes the whole class of problem at once:
every table is a table, every figure is placed, and the Word file and the PDF
are generated from the same sentences, so they cannot disagree.

    python scripts/update_paper_tex.py      # put the current numbers in the source
    python scripts/build_paper_docx.py      # -> 4.docx
    tectonic -X compile new.tex --outdir .  # -> new.pdf

Needs pandoc on PATH.
"""

import argparse
import re
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]


def flatten_authors(source):
    """Replace IEEEtran's author block with something pandoc can parse.

    pandoc's LaTeX reader does not know \\IEEEauthorblockN or the
    \\linebreakand this document defines on top of it, and it stops at the
    first one. The names and affiliation are recovered by pattern and written
    back as a plain \\author list. Only the copy handed to pandoc is changed;
    new.tex keeps its IEEE markup, which is what the PDF needs.
    """
    names = re.findall(r"\\IEEEauthorblockN\{([^}]*)\}", source)
    if not names:
        return source
    authors = " \\and ".join(
        f"{name}\\\\ PDPM IIITDM Jabalpur, India" for name in names)

    start = source.index("\\author{")
    depth, index = 0, start + len("\\author")
    while index < len(source):
        if source[index] == "{":
            depth += 1
        elif source[index] == "}":
            depth -= 1
            if depth == 0:
                break
        index += 1
    source = source[:start] + "\\author{" + authors + "}" + source[index + 1:]

    # The macro definition itself also confuses the reader once its uses are gone.
    return re.sub(r"\\makeatletter\s*\\newcommand\{\\linebreakand\}.*?\\makeatother\s*",
                  "", source, flags=re.S)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--tex", type=Path, default=REPO / "new.tex")
    parser.add_argument("--output", type=Path, default=REPO / "4.docx")
    args = parser.parse_args()

    if not shutil.which("pandoc"):
        raise SystemExit("pandoc is not on PATH")

    source = flatten_authors(args.tex.read_text())
    with tempfile.TemporaryDirectory() as workdir:
        staged = Path(workdir) / "paper.tex"
        staged.write_text(source)
        result = subprocess.run(
            ["pandoc", str(staged), "--from=latex", "--to=docx", "--standalone",
             f"--resource-path={REPO}:{REPO / 'figures'}",
             "-o", str(args.output)],
            capture_output=True, text=True,
        )
    if result.returncode != 0:
        sys.stderr.write(result.stdout + result.stderr)
        raise SystemExit(f"pandoc failed ({result.returncode})")
    if result.stderr.strip():
        print(result.stderr.strip())

    import docx
    document = docx.Document(str(args.output))
    print(f"wrote {args.output}: {len(document.paragraphs)} paragraphs, "
          f"{len(document.tables)} tables")


if __name__ == "__main__":
    main()
