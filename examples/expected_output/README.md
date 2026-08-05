# Expected example outputs

After downloading the trained checkpoint to `weights/best.pt`, run:

```bash
python inference/generate_probability_map.py
python inference/generate_damage_mask.py
```

Expected outputs are written to:

```text
examples/output/test_sample_probability.tif
examples/output/test_sample_damage_mask.tif
```

The sample verifies that the installation and inference pipeline execute correctly. It is not the complete PIG or Getz evaluation dataset.
