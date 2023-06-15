import numpy as np
import torch


def get_mAP(logits, gt_file, labellist, ap_file):
    if isinstance(logits, torch.Tensor):
        logits = logits.cpu().numpy()
    targets = np.zeros_like(logits)

    # When mapping labels from other datasets to our label list, there might be
    # more than one labels mapped to one of our label due to different
    # definitions. So there can be duplicate labels in `labellist`. This
    # special case is taken into account.
    label2idxs = {}
    for i, label in enumerate(labellist):
        if label not in label2idxs:
            label2idxs[label] = []
        label2idxs[label].append(i)

    with open(gt_file, "r") as f:
        lines = [line.strip("\n").split(",") for line in f.readlines()]
    assert len(lines) == targets.shape[0]
    for i, line in enumerate(lines):
        for tag in line[1:]:
            for idx in label2idxs[tag]:
                targets[i, idx] = 1.0

    _, APs = mAP_class(targets, logits)

    label2AP = dict()
    for label, AP in zip(labellist, APs):
        if label not in label2AP:
            label2AP[label] = AP
        else:
            assert (label2AP[label] - AP) < 0.001
    mAP = np.array(list(label2AP.values())).mean() * 100.0
    label2AP = sorted(list(label2AP.items()), key=lambda x: x[0])

    with open(ap_file, "w") as f:
        f.write("Tag,AP\n")
        for label, AP in label2AP:
            f.write(f"{label},{AP*100.0:.2f}\n")

    print(f"mAP: {mAP}")
    return mAP


def mAP_class(targs, preds):
    """Returns the model's average precision for each class
    Return:
        ap (FloatTensor): 1xK tensor, with avg precision for each class k
    """

    if np.size(preds) == 0:
        return 0
    ap = np.zeros((preds.shape[1]))
    # compute average precision for each class
    for k in range(preds.shape[1]):
        # sort scores
        scores = preds[:, k]
        targets = targs[:, k]
        # compute average precision
        ap[k] = average_precision(scores, targets)
    return 100 * ap.mean(), ap


def average_precision(output, target):
    epsilon = 1e-8

    # sort examples
    indices = output.argsort()[::-1]
    # Computes prec@i
    total_count_ = np.cumsum(np.ones((len(output), 1)))

    target_ = target[indices]
    ind = target_ == 1
    pos_count_ = np.cumsum(ind)
    total = pos_count_[-1]
    pos_count_[np.logical_not(ind)] = 0
    pp = pos_count_ / total_count_
    precision_at_i_ = np.sum(pp)
    precision_at_i = precision_at_i_ / (total + epsilon)

    return precision_at_i
