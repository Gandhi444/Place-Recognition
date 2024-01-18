import timm
import torch.linalg
from lightning import pytorch as pl
from pytorch_metric_learning import miners, losses, distances
from torchmetrics import MetricCollection

from zpo_project.metrics.multi import MultiMetric


class EmbeddingModel(pl.LightningModule):
    def __init__(self,
                 embedding_size: int,
                 lr: float,
                 lr_patience: int):
        super().__init__()

        self.save_hyperparameters()

        self.lr = lr
        self.lr_patience = lr_patience

        self.network = timm.create_model('resnet10t', pretrained=True, num_classes=embedding_size)

        # TODO: The distance, the miner and the loss function are subject to change
        # TODO: Adding embedding regularization is probably a good idea
        self.distance = distances.LpDistance(p=2)  # Euclidean distance
        self.miner = miners.MultiSimilarityMiner(distance=self.distance)
        self.loss_function = losses.TripletMarginLoss(distance=self.distance)

        self.val_outputs = None

        metrics = MetricCollection(MultiMetric(distance=self.distance))
        self.val_metrics = metrics.clone(prefix='val_')

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.network(x)

    def training_step(self, batch, batch_idx):
        x, y = batch
        x = x.squeeze(0)
        y = y.squeeze(0)
        y_pred = self.forward(x)
        loss = self.loss_function(y_pred, y, self.miner(y_pred, y))
        self.log('train_loss', loss, sync_dist=True, prog_bar=True)
        return loss

    def validation_step(self, batch, batch_idx) -> None:
        x, y = batch
        x = x.squeeze(0)
        y = y.squeeze(0)
        y_pred = self.forward(x)
        self.val_outputs['preds'].append(y_pred.cpu())
        self.val_outputs['targets'].append(y.cpu())

    def predict_step(self, batch, batch_idx, **kwargs) -> tuple[torch.Tensor, list[str]]:
        x, y = batch
        x = x.squeeze(0)
        y_pred = self.forward(x)
        return y_pred.cpu(), y

    def on_validation_epoch_start(self) -> None:
        self.val_outputs = {
            'preds': [],
            'targets': [],
        }

    def on_validation_epoch_end(self) -> None:
        preds = torch.cat(self.val_outputs['preds'], dim=0)
        targets = torch.cat(self.val_outputs['targets'], dim=0)
        self.log_dict(self.val_metrics(preds, targets), sync_dist=True)

    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(self.parameters(), lr=self.lr)
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=self.lr_patience)
        return {
            'optimizer': optimizer,
            'lr_scheduler': scheduler,
            'monitor': 'val_precision_at_1',
        }
