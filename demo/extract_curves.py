"""Run the trained detector on the held-out test split and dump curves as JSON."""
import json
import sys
from pathlib import Path

import numpy as np
from ultralytics import YOLO

root = Path(sys.argv[1])
out = Path(sys.argv[2])
model = YOLO(str(root / "runs/detect/v2/weights/best.pt"))
m = model.val(data=str(root / "dataset_v2/dataset.yaml"), split="test", imgsz=640,
              plots=True, verbose=False, workers=0, project=str(out.parent / "valruns"), name="t")
box = m.box
names = [m.names[i] for i in range(len(m.names))]
nc = len(names)
res = {"names": names, "curve_titles": list(m.curves), "n_images": 360}

# Per-class summary at the evaluation confidence (max-F1 point) and AP values
res["per_class"] = []
cls_ids = list(box.ap_class_index)
nt = getattr(m, "nt_per_class", None)
for k, ci in enumerate(cls_ids):
    p, r, ap50, ap = box.class_result(k)
    res["per_class"].append({"name": names[ci], "precision": float(p), "recall": float(r),
                             "ap50": float(ap50), "ap5095": float(ap),
                             "instances": int(nt[ci]) if nt is not None else None})
res["overall"] = {"precision": float(box.mp), "recall": float(box.mr),
                  "map50": float(box.map50), "map5095": float(box.map)}

# Curves: list of [x, y(nc,1000), xlabel, ylabel]
curves = []
for entry in m.curves_results:
    x, y, xl, yl = entry
    x = np.asarray(x)
    y = np.asarray(y)
    idx = np.linspace(0, len(x) - 1, 101).astype(int)
    curves.append({"xlabel": xl, "ylabel": yl, "x": [round(float(v), 4) for v in x[idx]],
                   "y": [[round(float(v), 4) for v in row[idx]] for row in y]})
res["curves"] = curves

cm = getattr(m, "confusion_matrix", None)
if cm is not None:
    res["confusion"] = np.asarray(cm.matrix).astype(int).tolist()
out.write_text(json.dumps(res))
print("ok", names, res["overall"])
for c in res["per_class"]:
    print(c)
print([c["xlabel"] + "/" + c["ylabel"] for c in curves])
