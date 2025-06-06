import json
from sklearn.metrics import classification_report
from sklearn.preprocessing import MultiLabelBinarizer

with open("outputs/batch_output.json") as f:
    preds = json.load(f)
with open("data/batch_input.json") as f:
    refs = json.load(f)

y_true = [r["labels"] for r in refs]
y_pred = [p["predicted"] for p in preds]

mlb = MultiLabelBinarizer()
mlb.fit(y_true + y_pred)

y_true_bin = mlb.transform(y_true)
y_pred_bin = mlb.transform(y_pred)

print(classification_report(y_true_bin, y_pred_bin, target_names=mlb.classes_))
