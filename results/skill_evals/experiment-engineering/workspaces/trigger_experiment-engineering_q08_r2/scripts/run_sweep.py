import itertools, json
from model import train_one

lrs = [1e-4, 3e-4, 1e-3]
widths = [256, 512, 1024]

results = []
for lr, width in itertools.product(lrs, widths):
    score = train_one(lr=lr, width=width, epochs=3)
    results.append({"lr": lr, "width": width, "score": score})

print(json.dumps(results))
