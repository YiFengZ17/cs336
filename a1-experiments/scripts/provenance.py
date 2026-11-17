"""Shared provenance utilities; importable by the experiment scripts."""
import sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[2]/"assignment1-basics"))
from cs336_basics.part4_training_loop.provenance import sha256, capture
