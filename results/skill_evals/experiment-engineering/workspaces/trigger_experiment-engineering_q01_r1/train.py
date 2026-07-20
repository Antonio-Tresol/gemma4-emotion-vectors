import torch
from model import build_model, get_loader

model = build_model().cuda()
opt = torch.optim.AdamW(model.parameters(), lr=3e-4)
loader = get_loader(batch_size=64)

for epoch in range(10):
    for batch in loader:
        opt.zero_grad()
        out = model(batch["input_ids"].cuda())
        loss = out.loss
        loss.backward()
        opt.step()
