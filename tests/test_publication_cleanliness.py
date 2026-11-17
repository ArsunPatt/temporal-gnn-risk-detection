import json
from pathlib import Path

def test_current_notebooks_have_no_obsolete_stage_instructions():
    forbidden = ["Next authorized stage", "Stop here", "Stop before Stage", "requires a new instruction", "Stage 4 still requires explicit authorization", "no Stage 5 work", "Stage 3 needs explicit authorization"]
    for path in Path("notebooks").glob("*.ipynb"):
        notebook = json.loads(path.read_text())
        text = json.dumps(notebook["cells"])
        for phrase in forbidden:
            assert phrase not in text, (path, phrase)
