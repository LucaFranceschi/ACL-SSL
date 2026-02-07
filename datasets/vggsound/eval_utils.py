import torch
import numpy as np
from sklearn import metrics as mt
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
        self.std_metrics = {}
        self.silence_metrics = {'pIA': [], 'metrics': {'AUC_N': 0.0, 'ap50': 0.0, 'pIA_hat': 0.0}}
        self.noise_metrics = {'pIA': [], 'metrics': {'AUC_N': 0.0, 'ap50': 0.0, 'pIA_hat': 0.0}}

    def evaluate_batch(self, heatmap: torch.Tensor, sil_heatmap: torch.Tensor, noise_heatmap: torch.Tensor,
                        thr: Optional[float] = None, **kwargs) -> None:
        """
        Evaluate a batch of predictions.

        Args:
            pred (torch.Tensor): Model predictions.
            thr (Optional[float]): Threshold for binary classification. If None, dynamically determined.

        Returns:
            None
        """
        for j in range(heatmap.size(0)):
            infer = heatmap[j]
            infer_s = sil_heatmap[j]
            infer_n = noise_heatmap[j]
            if thr is None:
                thr = np.sort(infer.detach().cpu().numpy().flatten())[int(infer.shape[1] * infer.shape[2] / 2)]
            self.cal_pIA(infer_s, infer_n, thr)

    def cal_pIA(self, infer_s: torch.Tensor, infer_n: torch.Tensor, thres: float = 0.01) -> List[float]:
        '''
        Calculate the percentage of Image Area as described in:
            Juanola, Xavier, et al. "Learning from Silence and Noise for Visual Sound Source Localization."

        :param self: Description
        '''
        infer_map_s = torch.zeros_like(infer_s)
        infer_map_s[infer_s >= thres] = 1

        infer_map_n = torch.zeros_like(infer_n)
        infer_map_n[infer_n >= thres] = 1

        shape = infer_map_n.shape

        pIA_s = torch.sum(infer_map_s.detach().cpu(), dim=(1, 2)).float() / (shape[1] * shape[2])
        pIA_n = torch.sum(infer_map_n.detach().cpu(), dim=(1, 2)).float() / (shape[1] * shape[2])

        self.silence_metrics['pIA'].append(pIA_s)
        self.noise_metrics['pIA'].append(pIA_n)

        return pIA_s, pIA_n

    def finalize_AUC_N(self):
        """
        Calculate the Area Under the Curve for Negative audio samples (AUC_N).

        Returns:
            float: AUC value.
        """
        for metrics in [self.silence_metrics, self.noise_metrics]:
            aucs = [np.sum(np.array(metrics['pIA']) >= 0.05 * i) / len(metrics['pIA'])
                    for i in range(21)]
            thr = [0.05 * i for i in range(21)]
            auc = mt.auc(thr, aucs)
            metrics['metrics']['AUC_N'] = auc

    def finalize_AP50(self):
        """
        Calculate Average Precision (piA@0.5).

        Returns:
            float: pIA@0.5 value.
        """
        for metrics in [self.silence_metrics, self.noise_metrics]:
            ap50 = np.mean(np.array(metrics['pIA']) <= 0.5)
            metrics['metrics']['ap50'] = ap50

    def finalize_pIA(self):
        """
        Calculate mean pIA.

        Returns:
            float: Mean pIA value.
        """
        for metrics in [self.silence_metrics, self.noise_metrics]:
            pIA_hat = np.mean(np.array(metrics['pIA']))
            metrics['metrics']['pIA_hat'] = pIA_hat

    def finalize(self):
        """
        Finalize and return evaluation metrics.

        Returns:
            Tuple[List[str], Dict[str, float]]: List of metric names and corresponding values.
        """
        self.finalize_AP50()
        self.finalize_AUC_N()
        self.finalize_pIA()
        return self.std_metrics, self.silence_metrics['metrics'], self.noise_metrics['metrics']
