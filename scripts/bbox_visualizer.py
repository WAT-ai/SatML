import os
import gradio as gr
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import numpy as np

# Load model and data once
from models.bbox_model import BBoxModel
from src.data_loader import create_bbox_dataset

# Load model
attrs_files = [f for f in os.listdir("logs") if f.endswith("_attrs.yaml")]
attrs_files.sort()
bbox_model = BBoxModel.load(f"logs/{attrs_files[-1]}")

dataset = create_bbox_dataset("./data/raw_data/STARCOP_test", bbox_model.max_boxes).take(50)

dataset = bbox_model.preprocess_dataset(dataset)
test_images = [x for x, _ in dataset]
y_true = np.array([y for _, y in dataset], dtype=np.float32)

predictions = bbox_model.predict(dataset.batch(1))
print("predictions:", predictions.shape)
print("test_images:", len(test_images))
print("y_true:", len(y_true))


def get_coords_from_bbox(bbox):
    """
    Converts objectness, x_center, y_center, width, height format to x_left, x_right, y_top, y_bottom.
    """
    x_center, y_center, width, height = bbox[1:5]
    x_left = x_center - width / 2
    x_right = x_center + width / 2
    y_top = y_center - height / 2
    y_bottom = y_center + height / 2
    return [x_left, x_right, y_top, y_bottom]


def grid_to_boxes(pred_grid, image_height, image_width, obj_thresh=0.5):
    """
    Convert YOLO-style grid output (S, S, B, 5) to bounding boxes in image coordinates.
    Each cell: [objectness, x_cell, y_cell, w, h] with x_cell/y_cell in [0, 1].
    """
    S, _, B, _ = pred_grid.shape
    boxes = []

    for i in range(S):
        for j in range(S):
            for b in range(B):
                obj_score, x_cell, y_cell, w, h = pred_grid[i, j, b]

                if obj_score < obj_thresh:
                    continue

                # Recover global normalized center coordinates
                x_center = (j + x_cell) / S
                y_center = (i + y_cell) / S

                x_abs = x_center * image_width
                y_abs = y_center * image_height
                w_abs = w * image_width
                h_abs = h * image_height

                x_left = x_abs - w_abs / 2
                x_right = x_abs + w_abs / 2
                y_top = y_abs - h_abs / 2
                y_bottom = y_abs + h_abs / 2

                boxes.append([x_left, x_right, y_top, y_bottom])

    return boxes



def draw_bounding_boxes(image: np.ndarray, ground_truth_boxes: list, predicted_boxes: list):
    """
    Draws ground truth and predicted bounding boxes on an image.

    Parameters:
    - image: 2D numpy array representing the image
    - ground_truth_boxes: List of [x_left, x_right, y_top, y_bottom] for ground truth
    - predicted_boxes: List of [x_left, x_right, y_top, y_bottom] for predictions
    """
    fig, ax = plt.subplots()
    ax.imshow(image, cmap="gray")  # Assuming grayscale image
    height, width = image.shape

    ground_truth_boxes = grid_to_boxes(ground_truth_boxes, height, width)
    predicted_boxes = grid_to_boxes(predicted_boxes, height, width, 0.5)

    print("Ground truth boxes:", ground_truth_boxes)
    print("Predicted boxes:", predicted_boxes)

    # Draw ground truth boxes in green
    for box in ground_truth_boxes:
        x_left, x_right, y_top, y_bottom = box
        width = x_right - x_left
        height = y_bottom - y_top
        rect = patches.Rectangle(
            (x_left, y_top),
            width,
            height,
            linewidth=2,
            edgecolor="green",
            facecolor="green",
            alpha=0.3,
        )  # Semi-transparent fill
        ax.add_patch(rect)

    # Draw predicted boxes in yellow
    for box in predicted_boxes:
        x_left, x_right, y_top, y_bottom = box
        width = x_right - x_left
        height = y_bottom - y_top
        rect = patches.Rectangle(
            (x_left, y_top),
            width,
            height,
            linewidth=2,
            edgecolor="yellow",
            facecolor="yellow",
            alpha=0.3,
        )  # Semi-transparent fill
        ax.add_patch(rect)

    ax.axis("off")  # Optional: hide axes
    return fig


def visualize(slice_idx):
    fig = draw_bounding_boxes(test_images[slice_idx][..., 0], y_true[slice_idx], predictions[slice_idx])

    return fig


if __name__ == "__main__":
    interface = gr.Interface(
        fn=visualize,
        inputs=gr.Slider(0, len(test_images) - 1, step=1, label="Select Slice"),
        outputs=gr.Plot(),
        live=True,
    )
    interface.launch()
