```bash
uv sync
```
Infer guide:

**For classification model (default):**

```bash
uv run python3 infer.py --image path/to/image.jpg --model model_full_17epoch.pth --output colorized.jpg
```

**For regression (baseline) model:**
```bash 
uv run python3 infer.py --image path/to/image.jpg --model model_l2_baseline.pth --model-type regression --output colorized.jpg
```