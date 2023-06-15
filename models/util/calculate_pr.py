import numpy as np
import logging
logger = logging.getLogger(__name__)
EPSILON = 1.e-9


def label_to_matrix(pred_gt_list, label_list, gt_col=3, pred_col=2):
    '''
        convert raw output to a numpy ndarray, the values of labels are:
        positive:1.0,
        negative: 0,
        Args:
            filename: raw output where columns are separeted by comma, and labels are
                separated by '+'
    '''
    label2idx = {tag: i for i, tag in enumerate(label_list)}
    label_num = len(pred_gt_list)
    y_true = np.zeros((label_num, len(label_list)))
    y_pred = np.zeros_like(y_true)
    for i in range(len(pred_gt_list)):
        line = pred_gt_list[i]
        gt_labels = []
        pred_labels = []
        if line[gt_col - 1]:
            gt_labels = line[gt_col - 1]
        if line[pred_col - 1]:
            pred_labels = line[pred_col - 1]
        for gt in gt_labels:
            if gt.strip('*') in label2idx:
                if '*' in gt:
                    idx = label2idx[gt.strip('*')]
                    y_true[i][idx] = 0.5
                else:
                    idx = label2idx[gt]
                    y_true[i][idx] = 1.0
            else:
                continue
                logger.warn('true label not in mapping:{}'.format(gt))
        for pred in pred_labels:
            if pred in label2idx:
                idx = label2idx[pred]
                y_pred[i][idx] = 1.0
            else:
                continue
                logger.warn('pred label not in mapping:{}'.format(pred))
    return y_true, y_pred, label_list

def precision_recall(y_true, y_pred, label_list=None, output=None):
    assert isinstance(y_true, np.ndarray), 'y_true should be np.array'
    assert isinstance(y_pred, np.ndarray), 'y_pred shold be np.array'
    assert y_true.shape == y_pred.shape, f'inconsistant shape, y_true:{y_true.shape}, y_pred:{y_pred.shape}'
    assert y_true.dtype == y_pred.dtype, f'inconsistant dtype, y_true:{y_true.dtype}, y_pred:{y_pred.dtype}'
    pred_true_sum = y_pred + y_true
    pred_true_subtract = y_pred - y_true
    tp = np.sum(pred_true_sum == 2, axis=0)
    fp = np.sum(pred_true_subtract == 1, axis=0)
    fn = np.sum(pred_true_subtract == -1, axis=0)
    precision = tp / (tp + fp + EPSILON) * 100
    recall = tp / (tp + fn + EPSILON) * 100
    class_precision = np.mean(precision)
    class_recall = np.mean(recall)
    if not label_list:
        label_list = range(len(tp))
    assert len(label_list) == len(tp), f'inconsistant length of label list:{len(label_list)} and tp:{len(tp)}'
    if output:
        with open(output, 'w', encoding="utf-8") as f:
            f.write('Tag,Precision,Recall\n')
            for i in range(len(tp)):
                f.write(f'{label_list[i]},{precision[i]:.2f},{recall[i]:.2f}\n')
    return class_precision, class_recall

def get_PR(path1,path2,label_list,result_file):
    #预测
    pred_dict = {}
    with open(path1,"r") as f:
        for line in f:
            line = line.strip().split(",")
            image_name = line[0]
            if len(line)>=2:
                labels = line[1:]
                pred_dict[image_name] = labels
            else:
                pred_dict[image_name] = []
    #标注
    annotation_dict = {}
    with open(path2,"r") as f:
        for line in f:
            line = line.strip().split(",")
            gt_image_name = line[0]
            annotation_dict[gt_image_name] = line[1:]
    pred_gt_list = []

    for k,v in annotation_dict.items():
        try:
            pred_gt_list.append([k,pred_dict[k],annotation_dict[k]])
        except:
            print("error image:",k,)
    print("length_pred:{}_gt:{}_pred&gt:{}".format(len(pred_dict),len(annotation_dict),len(pred_gt_list)))
    y_true, y_pred, output_label_list = label_to_matrix(pred_gt_list, label_list)
    CP, CR = precision_recall(y_true, y_pred, output_label_list, result_file)
    print(f"CP: {CP}")
    print(f"CR: {CR}")
    return CP, CR
