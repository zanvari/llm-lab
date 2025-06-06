import json
from scipy.stats import pearsonr, spearmanr

with open("outputs/batch_output.json") as f:
    preds = json.load(f)
with open("data/batch_input.json") as f:
    refs = json.load(f)

y_pred = [p["score"] for p in preds]
y_true = [r["label"] for r in refs]

print("Pearson:", pearsonr(y_pred, y_true)[0])
print("Spearman:", spearmanr(y_pred, y_true)[0])
