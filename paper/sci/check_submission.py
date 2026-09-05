"""Check this manuscript's Statistics submission requirements (standard library)."""

from __future__ import annotations

import argparse
from collections import Counter
from pathlib import Path
import re
import struct
import subprocess
import zipfile

HERE = Path(__file__).resolve().parent


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--compiled", action="store_true", help="also inspect LaTeX log and bibliography")
    parser.add_argument("--zip", type=Path, help="create a self-contained submission archive after checks")
    args = parser.parse_args()
    # Packaging must not silently reuse a stale PDF and a successful old log.
    # -g forces compilation; dependencies and auxiliary files stay local.
    if args.zip:
        subprocess.run(
            ["latexmk", "-g", "-pdf", "-interaction=nonstopmode", "-halt-on-error", "manuscript.tex"],
            cwd=HERE, check=True,
        )
    source = (HERE / "manuscript.tex").read_text(encoding="utf-8")
    tex = re.sub(r"(?<!\\)%[^\n]*", "", source)
    abstract = re.search(r"\\begin\{abstract\}(.*?)\\end\{abstract\}", tex, re.S).group(1)
    # Hyphenated terms and numeric ranges count as one word; SCI is one word.
    words = abstract.replace(r"\SCI", "SCI").replace(r"\%", "%").split()
    assert len(words) == 150, f"Abstract: {len(words)} words, expected 150"
    keywords = re.search(r"\\begin\{keywords\}(.*?)\\end\{keywords\}", tex, re.S).group(1)
    assert 3 <= len(keywords.split(";")) <= 5
    assert r"\documentclass{interact}" in tex and r"\bibliographystyle{tfnlm}" in tex
    assert tex.index(r"\bibliography{references}") < tex.index(r"\appendix")
    for heading in ("Funding", "Disclosure of interest", "Data availability statement",
                    "Declaration of generative AI use", "Notes on contributor"):
        assert rf"\section*{{{heading}}}" in tex, f"Missing {heading}"
    for obsolete in ("certified convexity", "certified concavity", "reliably convex",
                     "reliably concave", "sampled from a", "random beta design",
                     "cannot manufacture information"):
        assert obsolete not in tex.lower(), f"Outdated interpretation: {obsolete}"
    availability = re.search(r"\\section\*\{Data availability statement\}(.*?)\\section\*", tex, re.S).group(1)
    assert "https://doi.org/10.5281/zenodo.22338783" in availability, "Missing version-specific archive DOI"
    assert "sci-v1.0.0" in availability, "Missing archived release version"
    assert not re.search(r"\\cite[tp]?\{\*\}|\\nocite|\\citet\b", tex)

    labels = re.findall(r"\\label\{([^}]+)\}", tex)
    assert not [label for label, count in Counter(labels).items() if count > 1], "Duplicate labels"
    references = re.findall(r"\\(?:eqref|ref|autoref)\{([^}]+)\}", tex)
    assert set(references) <= set(labels), f"Undefined references: {set(references) - set(labels)}"
    # Remove complete floats so a caption/self-reference cannot pass this test.
    floats = list(re.finditer(r"\\begin\{(figure|table)\}(.*?)\\end\{\1\}", tex, re.S))
    body = tex
    for match in reversed(floats):
        body = body[:match.start()] + " " * len(match.group()) + body[match.end():]
    order = {"figure": [], "table": []}
    for match in floats:
        kind, content = match.group(1), match.group(2)
        float_labels = re.findall(r"\\label\{([^}]+)\}", content)
        assert len(float_labels) == 1 and r"\caption{" in content
        label = float_labels[0]
        callout = re.search(r"\\ref\{" + re.escape(label) + r"\}", body)
        assert callout, f"No body callout for {label}"
        assert callout.start() < match.start(), f"First callout follows float: {label}"
        order[kind].append(callout.start())
        print(f"PASS {kind} {label}: body callout precedes float")
    assert all(positions == sorted(positions) for positions in order.values()), "Callouts out of order"

    bib = (HERE / "references.bib").read_text(encoding="utf-8")
    keys = re.findall(r"@\w+\{([^,]+),", bib)
    cited = {key.strip() for group in re.findall(r"\\cite[tp]?\{([^}]+)\}", tex) for key in group.split(",")}
    assert len(keys) == len(set(keys)), "Duplicate bibliography keys"
    assert cited == set(keys), f"Citation mismatch: {cited.symmetric_difference(keys)}"
    graphics = re.findall(r"\\includegraphics\[width=([\d.]+)\\linewidth\]\{([^}]+)\}", tex)
    assert len(graphics) == len(order["figure"])
    for scale, name in graphics:
        path = HERE / name
        assert path.is_file(), f"Missing graphic: {name}"
        if path.suffix == ".png":
            # interact's default text width is 34 picas = 408 TeX points.
            width = struct.unpack(">I", path.read_bytes()[16:20])[0]
            dpi = width / (float(scale) * 408 / 72.27)
            assert dpi >= 300, f"Colour figure below 300 effective dpi: {name} ({dpi:.1f})"
            print(f"PASS {name}: {dpi:.1f} effective dpi")

    count_result = subprocess.run(
        ["texcount", "-inc", "-total", "-sum=1,1,1,0,0,0,0", "manuscript.tex"],
        cwd=HERE, capture_output=True, text=True, check=True,
    )
    count = re.search(r"Sum count: (\d+)", count_result.stdout).group(1)
    assert (HERE / "word_count.tex").read_text().strip() == count, f"Update word_count.tex to {count}"
    if args.compiled or args.zip:
        log = (HERE / "manuscript.log").read_text(errors="replace")
        assert not re.search(r"Warning:|Overfull|Undefined control sequence|^!", log, re.M), "Inspect manuscript.log"
        bbl = (HERE / "manuscript.bbl").read_text(encoding="utf-8")
        assert set(re.findall(r"\\bibitem\{([^}]+)\}", bbl)) == cited
        assert (HERE / "manuscript.pdf").is_file()
    print(f"PASS abstract: 150 words; keywords: {len(keywords.split(';'))}; bibliography: {len(keys)} cited entries; word count: {count}")

    if args.zip:
        files = ["manuscript.tex", "manuscript.pdf", "manuscript.bbl", "references.bib",
                 "word_count.tex", "interact.cls", "tfnlm.bst", "README.md",
                 "JOURNAL_PREPARATION.md", "TEMPLATE_PROVENANCE.md", "check_submission.py"]
        files += [name for _, name in graphics]
        destination = args.zip.resolve()
        destination.parent.mkdir(parents=True, exist_ok=True)
        # Fixed member order and timestamps make the source package reproducible.
        with zipfile.ZipFile(destination, "w", zipfile.ZIP_DEFLATED) as archive:
            for name in sorted(files):
                member = zipfile.ZipInfo(name, date_time=(2026, 9, 5, 0, 0, 0))
                member.compress_type = zipfile.ZIP_DEFLATED
                archive.writestr(member, (HERE / name).read_bytes())
        print(f"Wrote {destination}")


if __name__ == "__main__":
    main()
