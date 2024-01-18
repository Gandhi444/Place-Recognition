import pickle
from pathlib import Path

import lightning.pytorch as pl

from zpo_project.datamodules.metric_learning import MetricLearningDataModule
from zpo_project.models.model import EmbeddingModel


def train():
    pl.seed_everything(42, workers=True)

    # TODO: experiment with data module and model settings
    datamodule = MetricLearningDataModule(
        data_path=Path('data'),
        number_of_places_per_batch=8,
        number_of_images_per_place=2,
        number_of_batches_per_epoch=100,
        augment=False,
        validation_batch_size=16,
        number_of_workers=2
    )
    model = EmbeddingModel(
        embedding_size=1024,
        lr=3e-4,
        lr_patience=10
    )

    model_summary_callback = pl.callbacks.ModelSummary(max_depth=-1)
    checkpoint_callback = pl.callbacks.ModelCheckpoint(filename='{epoch}-{val_precision_at_1:.5f}', mode='max',
                                                       monitor='val_precision_at_1', verbose=True, save_last=True)
    early_stop_callback = pl.callbacks.EarlyStopping(monitor='val_precision_at_1', mode='max', patience=50)
    lr_monitor = pl.callbacks.LearningRateMonitor(logging_interval='epoch')

    trainer = pl.Trainer(
        callbacks=[model_summary_callback, checkpoint_callback, early_stop_callback, lr_monitor],
        accelerator='gpu',
        max_epochs=1000
    )

    trainer.fit(model=model, datamodule=datamodule)
    predictions = trainer.predict(model=model, ckpt_path=checkpoint_callback.best_model_path, datamodule=datamodule)

    results = {}
    for prediction in predictions:
        for embedding, identifier in zip(*prediction):
            results[identifier] = embedding.tolist()

    with open('results.pickle', 'wb') as file:
        pickle.dump(results, file)


if __name__ == '__main__':
    train()
