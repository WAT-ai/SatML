import os
import tensorflow as tf
import matplotlib.pyplot as plt
import matplotlib.patches as patches

from keras.src.callbacks import ModelCheckpoint, LearningRateScheduler, TensorBoard
from tensorflow.python.keras.models import load_model

from model import create_model
from data_loader import create_dataset 

def train_model():
    """
    Load data, preprocess it, and train the model.

    """
    dataset = create_dataset().batch(batch_size=16)
    # split into train and validation datasets
    train_dataset = dataset.take(18).repeat() 
    val_dataset = dataset.skip(18).take(17).repeat() 
    
    img_shape = 0
    for image_batch, label_batch in dataset.take(1):  
        # print("Batch shape:", image_batch.shape)  # shape of the entire batch
        # print("Image shape:", image_batch[0].shape)  # shape of one image in the batch
        img_shape = image_batch[0].shape
        # print("Label shape:", label_batch.shape)  # shape of the corresponding labels
        break 

    model = create_model(img_shape, 10)

    model.compile(
        optimizer='adam',
        loss='mse',
        # loss=tf.keras.losses.Huber(), 
        metrics=['mae', 'accuracy']
        # loss_weights= 0.5   
    )

    # callbacks
    checkpoint = ModelCheckpoint(
        filepath='best_model.keras',
        save_best_only=True,
        monitor='val_loss',
        mode='min'
    )

    lr_schedule = LearningRateScheduler(lambda epoch: 1e-3 * 0.95 ** epoch) # adjusts the learning rate for each epoch 
    # higher learning rate in the beginning helps the model converge faster.
    # lower learning rate later prevents overshooting the minimum and allows the model to fine-tune and be more precise.

    tensorboard = TensorBoard(log_dir='logs', histogram_freq=1) # logs various training metrics 
    # (like loss, accuracy, learning rate, etc.) during training so you can visualize them

    model.fit(
        train_dataset,
        epochs=20,
        steps_per_epoch=18,
        validation_data=val_dataset,
        validation_steps=17,
        callbacks=[checkpoint, lr_schedule]
    )

    prediction_dataset = dataset.take(1)

    for image, bbox in prediction_dataset:
        prediction = model.predict(image)
        print(f"Predicted bounding box: {prediction}")

    model.save('bounding_box_model.keras')
    print("Bounding box model saved successfully.")

if __name__ == "__main__":        
    train_model()
