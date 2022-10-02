from torch import Tensor
from sklearn import metrics

class BinaryMetric():
    def __init__(self, y: Tensor):
        self.y_true = y.cpu().data.numpy().flatten()
        
    def score2label(self, y_hat: Tensor):
        y_hat_numpy = y_hat.cpu().data.numpy().flatten()
        y_pred = [0 if y < 0.5 else 1 for y in y_hat_numpy]
        
        return y_pred
    
    def accuracy(self, y_hat: Tensor):
        y_pred = self.score2label(y_hat)
        return metrics.accuracy_score(self.y_true, y_pred)
    
    def recall(self, y_hat: Tensor):
        y_pred = self.score2label(y_hat)
        return metrics.recall_score(self.y_true, y_pred)
    
    def precision(self, y_hat: Tensor):
        y_pred = self.score2label(y_hat)
        return metrics.precision_score(self.y_true, y_pred)
    
    def f1_score(self, y_hat: Tensor):
        y_pred = self.score2label(y_hat)
        return metrics.f1_score(self.y_true, y_pred)
    
    def auroc(self, y_hat: Tensor):
        y_scores = y_hat.cpu().data.numpy().flatten()
        fpr, tpr, thresholds = metrics.roc_curve(self.y_true, y_scores)
        auroc_score = metrics.auc(fpr, tpr)
        return auroc_score

    def auprc(self, y_hat: Tensor):
        y_scores = y_hat.cpu().data.numpy().flatten()
        precision, recall, thresholds = metrics.precision_recall_curve(self.y_true, y_scores)
        auprc_score = metrics.auc(recall, precision)
        return auprc_score