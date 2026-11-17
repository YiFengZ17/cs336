"""One-time, idempotent migration; no training or downloads."""
from pathlib import Path
import datetime, hashlib, json, shutil, tarfile

HOME = Path.home()
ROOT = HOME / "projects/CS336"
DATA = HOME / "datasets/CS336"
RUNS = HOME / "experiments/CS336"
STAMP = datetime.datetime.now().strftime("%Y%m%d_%H%M%S_%f")
BACKUP = RUNS / "_maintenance" / STAMP
BACKUP.mkdir(parents=True)
with tarfile.open(BACKUP / "source-before.tar.gz", "w:gz") as archive:
    for base in [ROOT / "a1-experiments", ROOT / "assignment1-basics/cs336_basics/part4_training_loop"]:
        for p in base.rglob("*"):
            if p.is_file() and p.suffix in {".py", ".md", ".json"}:
                archive.add(p, arcname=str(p.relative_to(ROOT)), recursive=False)
    archive.add(ROOT / ".gitignore", arcname=".gitignore")

def digest(p):
    h = hashlib.sha256()
    with p.open("rb") as f:
        for chunk in iter(lambda: f.read(1024*1024), b""):
            h.update(chunk)
    return h.hexdigest()

def inventory(base):
    return {str(p.relative_to(base)): digest(p) for p in base.rglob("*") if p.is_file()}

moves=[]
def relocate(src, dst):
    if src.is_symlink():
        if src.resolve() != dst.resolve():
            raise RuntimeError(f"Unexpected existing link: {src}")
        return
    before=inventory(src) if src.exists() else {}
    dst.parent.mkdir(parents=True, exist_ok=True)
    if dst.exists():
        if before and inventory(dst) != before:
            raise RuntimeError(f"Destination differs: {dst}")
        if src.exists():
            shutil.move(str(src), str(BACKUP / ("redundant-"+src.name)))
    elif src.exists():
        shutil.move(str(src), str(dst))
    else:
        dst.mkdir()
    if before and inventory(dst) != before:
        raise RuntimeError(f"Hash mismatch: {dst}")
    src.parent.mkdir(parents=True, exist_ok=True)
    src.symlink_to(dst, target_is_directory=True)
    moves.append({"source":str(src), "destination":str(dst), "sha256":before})

old=ROOT / "a1-experiments"
relocate(old/"data/demo", DATA/"demo/v1")
relocate(old/"data/tinystories_small", DATA/"tinystories/processed/bpe1000-small-v1")
# Parent data becomes a compatibility view; new corpus variants live outside the repository.
for name,target in [("tinystories_full",DATA/"tinystories/processed/bpe10000-v1"),("owt_full",DATA/"owt/processed/bpe32000-v1"),("owt_small",DATA/"owt/processed/bpe1000-small-v1")]:
    link=old/"data"/name
    if not link.exists() and not link.is_symlink(): link.symlink_to(target, target_is_directory=True)
for name in ["checkpoints","logs","results"]:
    relocate(old/name, RUNS/"a1/legacy"/name if name != "results" else RUNS/"a1/suites")
for name in ["data","checkpoints"]:
    relocate(ROOT/name, RUNS/"a1/legacy"/("root-"+name))
for name in ["a2","a3","a4","a5"]:
    (RUNS/name).mkdir(parents=True,exist_ok=True)
for name in ["tinystories/raw", "owt/raw", "a4", "a5"]:
    (DATA/name).mkdir(parents=True,exist_ok=True)
(HOME/"models/CS336").mkdir(parents=True,exist_ok=True)
for base in [DATA/"demo/v1",DATA/"tinystories/processed/bpe1000-small-v1"]:
    manifest=base/"dataset-manifest.json"
    files={str(p.relative_to(base)):{"bytes":p.stat().st_size,"sha256":digest(p)} for p in base.rglob("*") if p.is_file() and p != manifest}
    manifest.write_text(json.dumps({"version":base.name,"origin":"Migrated existing CS336 artifacts; original download revision not recorded", "files":files},indent=2)+"\n")
(BACKUP/"migration.json").write_text(json.dumps(moves,indent=2)+"\n")
print(f"Migrated and hash-verified {sum(len(m['sha256']) for m in moves)} files; backup: {BACKUP}")
