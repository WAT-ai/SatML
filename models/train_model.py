from keras.src.callbacks import ModelCheckpoint, LearningRateScheduler, TensorBoard

from models.model import create_model
from src.data_loader import create_bbox_dataset, augment_dataset
from src.image_utils import compare_bbox

def train_model(data_dir: str = './data/raw_data/STARCOP_train_easy', max_boxes=10):
    """
    Load data, preprocess it, and train the model.

    """
    dataset = create_bbox_dataset(data_dir, max_boxes=max_boxes)
    dataset = dataset.flat_map(augment_dataset)
    dataset = dataset.batch(batch_size=8)
    train_dataset = dataset.take(50).repeat()
    
    img_shape = 0
    for image_batch, _ in dataset.take(1):
        img_shape = image_batch.shape[1:]
        break 

    model = create_model(img_shape, max_boxes)

    model.compile(
        optimizer='adam',
        loss=compare_bbox,
        metrics=['mae', 'accuracy']
        # loss_weights= 0.5   
    )

    # callbacks
    checkpoint = ModelCheckpoint(
        filepath='bounding_box_model_epoch_{epoch:02d}.keras',
        save_best_only=False,
        monitor='val_loss',
        mode='min',
        save_freq='epoch'  # Save model after each epoch
    )

    lr_schedule = LearningRateScheduler(lambda epoch: 1e-3 * 0.95 ** epoch) # adjusts the learning rate for each epoch 
    # higher learning rate in the beginning helps the model converge faster.
    # lower learning rate later prevents overshooting the minimum and allows the model to fine-tune and be more precise.

    tensorboard = TensorBoard(log_dir='logs', histogram_freq=1) # logs various training metrics 
    # (like loss, accuracy, learning rate, etc.) during training so you can visualize them

    model.fit(
        train_dataset,
        epochs=10,  # Increased epochs to see model saving after each epoch
        steps_per_epoch=50,
        callbacks=[checkpoint, lr_schedule, tensorboard]
    )

    one_batch = dataset.skip(50).take(1)
    for image, real_bbox in one_batch:
        print(f"Real bounding box: {real_bbox}")

    predict_dataset = dataset.skip(50).take(1)
    for image, _ in predict_dataset:
        prediction = model.predict(image)
        print(f"Predicted bounding box: {prediction}")

    model.save('bounding_box_model.keras')
    print("Bounding box model saved successfully.")

if __name__ == "__main__":
    train_model(max_boxes=1)
