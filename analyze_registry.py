"""Analyze model_registry.json for bloat."""
import json
from pathlib import Path

reg = json.loads(Path("models/model_registry.json").read_text())
print(f"Total versions: {len(reg)}")
print(f"Total file size: {Path('models/model_registry.json').stat().st_size:,} bytes")

# Check if dataset_info and feature_config are identical across all entries
dataset_infos = set()
feature_configs = set()
for v, entry in reg.items():
    dataset_infos.add(json.dumps(entry.get("dataset_info"), sort_keys=True))
    feature_configs.add(json.dumps(entry.get("feature_config"), sort_keys=True))

print(f"Unique dataset_info structures: {len(dataset_infos)}")
print(f"Unique feature_config structures: {len(feature_configs)}")

# Size of one entry's duplicated metadata
sample = list(reg.values())[0]
dup_size = len(json.dumps(sample.get("dataset_info", {}))) + len(json.dumps(sample.get("feature_config", {})))
print(f"Duplicated metadata per entry: ~{dup_size:,} chars")
print(f"Total duplicated: ~{dup_size * len(reg):,} chars ({dup_size * len(reg) / 1024:.0f} KB)")
