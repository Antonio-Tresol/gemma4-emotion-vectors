import json
import random
from pathlib import Path


def load_jsonl(path):
    return [json.loads(line) for line in Path(path).read_text().splitlines() if line]


def set_seed(seed=0):
    random.seed(seed)
