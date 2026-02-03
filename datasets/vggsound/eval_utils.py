import torch
import numpy as np
from sklearn import metrics
from typing import List, Optional, Tuple, Dict


class Evaluator(object):
    def __init__(self) -> None:
        """
        Initialize the VGG-Sound (VGGS) Evaluator.

        Attributes:
            PIA (List[float]): Buffer of Percentage of Image Area values.
            AUC_N (List[float]): Buffer of AUC_N values.
            N (int): Counter for the number of evaluations.
            metrics (List[str]): List of metric names.
        """
        super(Evaluator, self).__init__()
        self.pIA = []
        self.N = 0
        self.metrics = ['pIA', 'AUC_N']

    def evaluate_batch(self, pred: torch.Tensor, thr: Optional[float] = None) -> None:
        """
        Evaluate a batch of predictions against ground truth.

        Args:
            pred (torch.Tensor): Model predictions.
            thr (Optional[float]): Threshold for binary classification. If None, dynamically determined.

        Returns:
            None
        """
        for j in range(pred.size(0)):
            infer = pred[j]
            if thr is None:
                thr = np.sort(infer.detach().cpu().numpy().flatten())[int(infer.shape[1] * infer.shape[2] / 2)]
            self.cal_pIA(infer, thr)

    def cal_pIA(self, infer: torch.Tensor, thres: float = 0.01) -> List[float]:
        '''
        Calculate the percentage of Image Area as described in:
            Juanola, Xavier, et al. "Learning from Silence and Noise for Visual Sound Source Localization."

        :param self: Description
        '''
        infer_map = torch.zeros_like(infer)
        infer_map[infer >= thres] = 1
        shape = infer_map.shape

        pIA = (torch.sum(infer_map.detach().cpu(), dim=(1, 2)) / (shape[1] * shape[2])).tolist()

        self.pIA.append(pIA)

        return pIA

    def finalize_AUC_N(self) -> float:
        """
        Calculate the Area Under the Curve for Negative audio samples (AUC_N).

        Returns:
            float: AUC value.
        """
        aucs = [np.sum(np.array(self.pIA) >= 0.05 * i) / len(self.pIA)
                 for i in range(21)]
        thr = [0.05 * i for i in range(21)]
        auc = metrics.auc(thr, aucs)
        return auc

    def finalize_AP50(self) -> float:
        """
        Calculate Average Precision (piA@0.5).

        Returns:
            float: pIA@0.5 value.
        """
        ap50 = np.mean(np.array(self.pIA) <= 0.5)
        return ap50

    def finalize_pIA(self) -> float:
        """
        Calculate mean pIA.

        Returns:
            float: Mean pIA value.
        """
        pIA = np.mean(np.array(self.pIA))
        return pIA

    def finalize(self) -> Tuple[List[str], Dict[str, float]]:
        """
        Finalize and return evaluation metrics.

        Returns:
            Tuple[List[str], Dict[str, float]]: List of metric names and corresponding values.
        """
        ap50 = self.finalize_AP50() * 100
        auc = self.finalize_AUC_N() * 100
        return self.metrics, {self.metrics[0]: ap50, self.metrics[1]: auc}
