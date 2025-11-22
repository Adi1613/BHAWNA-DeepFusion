import numpy as np, torch, torch.nn as nn
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, average_precision_score, confusion_matrix,
    balanced_accuracy_score, cohen_kappa_score, matthews_corrcoef,
    log_loss, top_k_accuracy_score, classification_report,
)

def train_epoch(model, loader, opt, device, lam_va: float = 0.5):
    model.train()
    ce = nn.CrossEntropyLoss().to(device)
    l1 = nn.SmoothL1Loss().to(device)
    total = 0.0; seen = 0
    for batch in loader:
        if batch is None: continue
        vids, mels, eegs, y = batch
        vids = vids.to(device) if vids is not None else None
        mels = mels.to(device) if mels is not None else None
        eegs = eegs.to(device) if eegs is not None else None
        y = y.to(device)
        logits, va = model(vids, mels, eegs)
        loss = ce(logits, y)
        opt.zero_grad(); loss.backward(); opt.step()
        bs = y.size(0); total += loss.item()*bs; seen += bs
    return total/max(1,seen)


def eval_epoch_metrics(model, loader, device, n_classes: int, topk: int = 3):
    model.eval(); y_true=[]; y_pred=[]; y_proba=[]
    with torch.no_grad():
        for batch in loader:
            if batch is None: continue
            vids, mels, eegs, y = batch
            vids = vids.to(device) if vids is not None else None
            mels = mels.to(device) if mels is not None else None
            eegs = eegs.to(device) if eegs is not None else None
            y = y.to(device)
            logits, _ = model(vids, mels, eegs)
            probs = torch.softmax(logits, dim=1)
            pred = probs.argmax(dim=1)
            y_true.append(y.cpu().numpy()); y_pred.append(pred.cpu().numpy()); y_proba.append(probs.cpu().numpy())
    y_true = np.concatenate(y_true) if y_true else np.array([])
    y_pred = np.concatenate(y_pred) if y_pred else np.array([])
    y_proba= np.concatenate(y_proba) if y_proba else np.empty((0,n_classes))
    if y_true.size==0:
        return {"accuracy":0.0}
    labels = list(range(n_classes))
    out = {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "balanced_accuracy": float(balanced_accuracy_score(y_true, y_pred)),
        "f1_macro": float(f1_score(y_true, y_pred, average="macro", zero_division=0)),
        "f1_weighted": float(f1_score(y_true, y_pred, average="weighted", zero_division=0)),
        "confusion_matrix": confusion_matrix(y_true, y_pred, labels=labels).tolist(),
        "report": classification_report(y_true, y_pred, output_dict=True, zero_division=0),
    }
    if y_proba.size:
        out["roc_auc_ovr"] = float(roc_auc_score(y_true, y_proba, multi_class="ovr"))
        out["top_k_accuracy"] = float(top_k_accuracy_score(y_true, y_proba, k=min(topk,n_classes)))
    out["_y_true"], out["_y_pred"] = y_true, y_pred
    return out
