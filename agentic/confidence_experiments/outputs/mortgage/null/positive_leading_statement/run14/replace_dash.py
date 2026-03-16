from pathlib import Path
path = Path("write_conclusion.py")
text = path.read_text(encoding="utf-8")
text = text.replace("–", "-")
path.write_text(text, encoding="utf-8")
