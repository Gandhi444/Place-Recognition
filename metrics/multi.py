import logging

import torch
from pytorch_metric_learning.distances import BaseDistance
from pytorch_metric_learning.utils.accuracy_calculator import AccuracyCalculator
from pytorch_metric_learning.utils.inference import CustomKNN
from torchmetrics import Metric


class MultiMetric(Metric):
    def __init__(self, distance: BaseDistance):
        super().__init__()

        logger = logging.getLogger('PML')
        logger.setLevel(logging.WARN)

        knn = CustomKNN(distance, batch_size=256)
        self.calculator = AccuracyCalculator(include=('precision_at_1', 'mean_average_precision'), k=4,
                                             device=torch.device('cpu'),
                                             knn_func=knn)
        self.metric_names = self.calculator.get_curr_metrics()

        for metric_name in self.metric_names:
            self.add_state(metric_name, default=torch.tensor(0, dtype=torch.float32), dist_reduce_fx='sum')

        self.add_state('count', default=torch.tensor(0, dtype=torch.int64), dist_reduce_fx='sum')

    def update(self, vectors, labels):
        vectors = vectors.detach().cpu() if vectors.requires_grad else vectors.cpu()
        labels = labels.detach().cpu() if labels.requires_grad else labels.cpu()
        results = self.calculator.get_accuracy(vectors, labels, include=('precision_at_1', 'mean_average_precision'))
        for metric_name, metric_value in results.items():
            metric_state = getattr(self, metric_name)
            metric_state += metric_value

        self.count += 1

    def compute(self):
        return {
            metric_name: getattr(self, metric_name) / self.count for metric_name in self.metric_names
        }
