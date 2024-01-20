import pickle
from pathlib import Path

import lightning.pytorch as pl
import json
from datamodules.metric_learning import MetricLearningDataModule
from models.model import EmbeddingModel
import check

def train():
    pl.seed_everything(42, workers=True)
    f = open('keys.json')
    keys=json.load(f)
    neptune_key=keys['neptune_api']
    index=keys['index']
    f.close()
    # TODO: experiment with data module and model settings
    num_places=7
    num_images=3
    datamodule = MetricLearningDataModule(
        data_path=Path('data'),
        number_of_places_per_batch=num_places,
        number_of_images_per_place=num_images,
        number_of_batches_per_epoch=100,
        augment=True,
        validation_batch_size=num_places*num_images,
        number_of_workers=8
    )

    model = EmbeddingModel(
        embedding_size=1024,
        lr=3e-4,
        lr_patience=10,
        model_name='mobilenetv3_large_100.ra_in1k',
        l2norm=0.001
    )

    model_summary_callback = pl.callbacks.ModelSummary(max_depth=-1)
    checkpoint_callback = pl.callbacks.ModelCheckpoint(filename='{epoch}-{val_precision_at_1:.5f}', mode='max',
                                                       monitor='val_precision_at_1', verbose=True, save_last=True)
    early_stop_callback = pl.callbacks.EarlyStopping(monitor='val_precision_at_1', mode='max', patience=10)
    lr_monitor = pl.callbacks.LearningRateMonitor(logging_interval='epoch')
    
    logger = pl.loggers.NeptuneLogger(project='gandhi444/Place-Recognition',
                                  api_token=neptune_key,log_model_checkpoints=False)
    trainer = pl.Trainer(logger=logger,
        callbacks=[model_summary_callback, checkpoint_callback, early_stop_callback, lr_monitor],
        accelerator='gpu',
        max_epochs=50
    )

    trainer.fit(model=model, datamodule=datamodule)
    predictions = trainer.predict(model=model, ckpt_path=checkpoint_callback.best_model_path, datamodule=datamodule)

    results = {}
    for prediction in predictions:
        for embedding, identifier in zip(*prediction):
            results[identifier] = embedding.tolist()

    with open('results.pickle', 'wb') as file:
        pickle.dump(results, file)

    test_res=check.test(index,'cosine')
    if test_res is not None:
        logger.experiment["test_accuracy@1"] = test_res
        
    logger.finalize("END")
if __name__ == '__main__':
    train()
