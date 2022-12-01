import torch
import pytorch_lightning as pl
from pytorch_lightning import loggers as pl_loggers
from pytorch_lightning.callbacks import ModelCheckpoint
from src.utils.preprocess_utils import preprocess_fer2013
from src.multi_models.resnet.model import resnet, EmotionClassifier

processed_fer2013 = preprocess_fer2013('./datasets/FER2013/fer2013_1.csv')
resnet50 = resnet('./model_results/resnet_weights/resnet50_scratch_weight.pkl')

emotion_classifier = EmotionClassifier(model=resnet50, processed_fer2013=processed_fer2013, batch_size=64)
trainer = pl.Trainer(
    logger=pl_loggers.CSVLogger(
        save_dir='./model_results/resnet_runs/'
    ),
    callbacks=[
        ModelCheckpoint(
            dirpath='./model_results/resnet_runs/',
            monitor='val_accuracy',
            filename='fer2013_{val_accuracy:.3f}',
            mode='max'
        ),
    ],
    gpus=1,
    max_epochs=50,
)
trainer.fit(emotion_classifier)
trainer.test(emotion_classifier)