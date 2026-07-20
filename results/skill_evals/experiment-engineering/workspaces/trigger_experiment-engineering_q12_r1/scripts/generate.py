import json
from api_client import complete

prompts = [json.loads(l) for l in open("data/probe_prompts.jsonl")]
outputs = []
for p in prompts:
    outputs.append({"id": p["id"], "completion": complete(p["prompt"])})

with open("outputs.json", "w") as f:
    json.dump(outputs, f)
