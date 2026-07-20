import json

with open("results/preds.jsonl") as f:
    preds = [json.loads(l) for l in f]

acc = sum(p["pred"] == p["label"] for p in preds) / len(preds)
print(json.dumps({"accuracy": acc}))
