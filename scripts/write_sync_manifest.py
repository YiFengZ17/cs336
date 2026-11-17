"""Capture local provenance before transferring code without Git metadata."""
from pathlib import Path
import hashlib,json,subprocess
ROOT=Path(__file__).resolve().parents[1]
def git(base,*args):
    r=subprocess.run(["git","-C",str(base),*args],capture_output=True,text=True)
    return r.stdout.strip() if r.returncode==0 else "unavailable"
repos=[ROOT,*sorted(ROOT.glob("assignment*-*"))]
state={str(p.relative_to(ROOT)) or ".": {"commit":git(p,"rev-parse","HEAD"),"status":git(p,"status","--porcelain")} for p in repos if p.is_dir()}
hashes={}
for base in [ROOT/"scripts",ROOT/"a1-experiments",ROOT/"assignment1-basics/cs336_basics",ROOT/"docs"]:
    import os
    for directory,dirs,files in os.walk(base,followlinks=False):
        dirs[:]=[d for d in dirs if not (Path(directory)/d).is_symlink() and d not in {"data","checkpoints","logs","results","__pycache__","plans"}]
        for name in files:
            p=Path(directory)/name
            if p.suffix in {".py",".sh",".json",".md"}:
                hashes[str(p.relative_to(ROOT))]=hashlib.sha256(p.read_bytes()).hexdigest()
(ROOT/"code-sync-manifest.json").write_text(json.dumps({"source":"local WSL working tree","repositories":state,"sha256":hashes},ensure_ascii=False,indent=2)+"\n")
