"""Bundle all processed JSON files into a single JS payload for the dashboard.

Writing the data as a window global (instead of fetch-ing JSON) lets the
dashboard work both on GitHub Pages and when opened as a local file.
"""
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
PROCESSED = ROOT / "data" / "processed"
OUT = ROOT / "dashboard" / "data" / "data.js"

payload = {
    p.stem: json.loads(p.read_text(encoding="utf-8"))
    for p in sorted(PROCESSED.glob("*.json"))
}
OUT.parent.mkdir(parents=True, exist_ok=True)
OUT.write_text("window.WC = " + json.dumps(payload, ensure_ascii=False) + ";",
               encoding="utf-8")
print(f"wrote {OUT.relative_to(ROOT)} ({OUT.stat().st_size // 1024} KB)")
