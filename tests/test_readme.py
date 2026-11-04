"""The README's "Use it as a library" snippet must keep running."""

import re
from pathlib import Path

README = Path(__file__).resolve().parents[1] / "README.md"


def test_readme_library_snippet_runs(capsys):
    text = README.read_text(encoding="utf-8")
    block = re.search(
        r"<!-- snippet:start -->\s*```python\n(.*?)```\s*<!-- snippet:end -->", text, re.S
    )
    assert block, "README is missing the marked library snippet"
    code = block.group(1)
    assert len(code.strip().splitlines()) <= 15
    exec(compile(code, "README.md", "exec"), {})
    out = capsys.readouterr().out.split()
    assert float(out[0]) < 1e-8  # trim residual
    assert out[-1] == "False"  # no loss of control
