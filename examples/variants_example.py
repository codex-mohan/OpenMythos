from open_mythos import (
    mythos_3b,
    OpenMythos,
)

cfg = mythos_3b()
model = OpenMythos(cfg)

total = sum(p.numel() for p in model.parameters())
print(f"Parameters: {total:,}")
