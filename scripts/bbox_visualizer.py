import os
import argparse
import gradio as gr
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import numpy as np

# Load model and data once
from models.bbox_model import BBoxModel
from src.data_loader import create_bbox_dataset


def load_latest_model(log_dir="logs", model_prefix="bbox_model"):
    """Load the most recently saved BBoxModel."""
    attrs_files = [f for f in os.listdir(log_dir) if f.endswith("_attrs.yaml")]

    # Filter for bbox_model files specifically
    model_attrs = [
        f
        for f in attrs_files
        if model_prefix in f or "_bbox_model.keras" in f.replace("_attrs.yaml", "_bbox_model.keras")
    ]

    if not model_attrs:
        # If no specific model files, try to find any recent model files
        print("No bbox_model files found, looking for any model files...")
        attrs_files.sort(key=lambda x: os.path.getctime(os.path.join(log_dir, x)), reverse=True)
        if not attrs_files:
            raise FileNotFoundError("No model files found in logs directory")
        latest_file = attrs_files[0]
    else:
        model_attrs.sort(key=lambda x: os.path.getctime(os.path.join(log_dir, x)), reverse=True)
        latest_file = model_attrs[0]

    print(f"Loading model: {latest_file}")
    return BBoxModel.load(f"{log_dir}/{latest_file}")


def get_coords_from_bbox(bbox):
    """
    Converts objectness, x_center, y_center, width, height format to x_left, x_right, y_top, y_bottom.
    """
    if len(bbox) < 5:
        return None
    objectness, x_center, y_center, width, height = bbox[:5]
    if objectness < 0.1:  # Skip very low confidence
        return None
    x_left = x_center - width / 2
    x_right = x_center + width / 2
    y_top = y_center - height / 2
    y_bottom = y_center + height / 2
    return [x_left, x_right, y_top, y_bottom, objectness]


def grid_to_boxes(pred_grid, image_height, image_width, obj_thresh=0.5):
    """
    Convert YOLO-style grid output (S, S, B, 5) to bounding boxes in image coordinates.
    Each cell: [objectness, x_cell, y_cell, w, h] with x_cell/y_cell in [0, 1].

    Args:
        pred_grid: (S, S, B, 5) prediction grid from YOLO model
        image_height: Height of the image in pixels
        image_width: Width of the image in pixels
        obj_thresh: Objectness threshold for filtering detections

    Returns:
        List of [x_left, x_right, y_top, y_bottom, confidence] boxes
    """
    if len(pred_grid.shape) != 4:
        print(f"Warning: Expected 4D grid (S, S, B, 5), got shape {pred_grid.shape}")
        return []

    S, _, B, _ = pred_grid.shape
    boxes = []

    for i in range(S):
        for j in range(S):
            for b in range(B):
                if pred_grid[i, j, b, 0] < obj_thresh:  # objectness check
                    continue

                obj_score, x_cell, y_cell, w, h = pred_grid[i, j, b]

                # Recover global normalized center coordinates
                x_center = (j + x_cell) / S
                y_center = (i + y_cell) / S

                # Convert to absolute coordinates
                x_abs = x_center * image_width
                y_abs = y_center * image_height
                w_abs = w * image_width
                h_abs = h * image_height

                # Convert to corner coordinates
                x_left = x_abs - w_abs / 2
                x_right = x_abs + w_abs / 2
                y_top = y_abs - h_abs / 2
                y_bottom = y_abs + h_abs / 2

                boxes.append([x_left, x_right, y_top, y_bottom, obj_score])

    return boxes


def non_max_suppression(boxes, iou_threshold=0.5):
    """
    Apply Non-Maximum Suppression to remove overlapping detections.

    Args:
        boxes: List of [x_left, x_right, y_top, y_bottom, confidence]
        iou_threshold: IoU threshold for suppression

    Returns:
        List of boxes after NMS
    """
    if len(boxes) == 0:
        return []

    # Sort by confidence (highest first)
    boxes = sorted(boxes, key=lambda x: x[4], reverse=True)

    def calculate_iou(box1, box2):
        x1_left, x1_right, y1_top, y1_bottom = box1[:4]
        x2_left, x2_right, y2_top, y2_bottom = box2[:4]

        # Calculate intersection
        x_left = max(x1_left, x2_left)
        x_right = min(x1_right, x2_right)
        y_top = max(y1_top, y2_top)
        y_bottom = min(y1_bottom, y2_bottom)

        if x_right <= x_left or y_bottom <= y_top:
            return 0.0

        intersection = (x_right - x_left) * (y_bottom - y_top)
        area1 = (x1_right - x1_left) * (y1_bottom - y1_top)
        area2 = (x2_right - x2_left) * (y2_bottom - y2_top)
        union = area1 + area2 - intersection

        return intersection / union if union > 0 else 0.0

    final_boxes = []
    while boxes:
        # Take the highest confidence box
        best_box = boxes.pop(0)
        final_boxes.append(best_box)

        # Remove boxes with high IoU with the best box
        boxes = [box for box in boxes if calculate_iou(best_box, box) < iou_threshold]

    return final_boxes


def draw_bounding_boxes(
    image: np.ndarray,
    ground_truth_boxes: list,
    predicted_boxes: list,
    obj_thresh=0.5,
    nms_thresh=0.5,
    show_confidence=True,
):
    """
    Draws ground truth and predicted bounding boxes on an image.

    Parameters:
    - image: 2D numpy array representing the image
    - ground_truth_boxes: YOLO grid format (S, S, B, 5) for ground truth
    - predicted_boxes: YOLO grid format (S, S, B, 5) for predictions
    - obj_thresh: Objectness threshold for filtering detections
    - nms_thresh: NMS IoU threshold
    - show_confidence: Whether to display confidence scores
    """
    fig, ax = plt.subplots(figsize=(12, 8))
    ax.imshow(image, cmap="gray")
    height, width = image.shape

    # Convert grid predictions to bounding boxes
    gt_boxes = grid_to_boxes(ground_truth_boxes, height, width, obj_thresh=0.1)  # Lower threshold for GT
    pred_boxes = grid_to_boxes(predicted_boxes, height, width, obj_thresh=obj_thresh)

    # Apply NMS to predictions
    pred_boxes = non_max_suppression(pred_boxes, iou_threshold=nms_thresh)

    print(f"Ground truth boxes: {len(gt_boxes)}")
    print(f"Predicted boxes (after NMS): {len(pred_boxes)}")

    # Draw ground truth boxes in green
    for box in gt_boxes:
        x_left, x_right, y_top, y_bottom = box[:4]
        box_width = x_right - x_left
        box_height = y_bottom - y_top
        rect = patches.Rectangle(
            (x_left, y_top),
            box_width,
            box_height,
            linewidth=3,
            edgecolor="green",
            facecolor="green",
            alpha=0.2,
            label="Ground Truth" if box == gt_boxes[0] else "",
        )
        ax.add_patch(rect)

    # Draw predicted boxes in red
    for i, box in enumerate(pred_boxes):
        x_left, x_right, y_top, y_bottom, confidence = box
        box_width = x_right - x_left
        box_height = y_bottom - y_top
        rect = patches.Rectangle(
            (x_left, y_top),
            box_width,
            box_height,
            linewidth=2,
            edgecolor="red",
            facecolor="red",
            alpha=0.3,
            label="Prediction" if i == 0 else "",
        )
        ax.add_patch(rect)

        # Add confidence score text
        if show_confidence:
            ax.text(
                x_left,
                y_top - 5,
                f"{confidence:.2f}",
                color="red",
                fontsize=10,
                fontweight="bold",
                bbox=dict(boxstyle="round,pad=0.2", facecolor="white", alpha=0.8),
            )

    ax.set_title(f"YOLO Model Predictions (Threshold: {obj_thresh:.2f})", fontsize=14)
    ax.legend()
    ax.axis("off")

    return fig


class YOLOVisualizer:
    def __init__(self, model_path=None, data_dir="./data/raw_data/STARCOP_test", max_samples=50):
        # Load model
        if model_path:
            self.bbox_model = BBoxModel.load(model_path)
        else:
            self.bbox_model = load_latest_model()

        print(
            f"Loaded model with grid size: {self.bbox_model.grid_size}, boxes per grid: {self.bbox_model.boxes_per_grid}"
        )

        # Load dataset
        self.dataset = create_bbox_dataset(data_dir, self.bbox_model.max_boxes).take(max_samples)
        self.dataset = self.bbox_model.preprocess_dataset(self.dataset)

        # Prepare data
        self.test_images = [x for x, _ in self.dataset]
        self.y_true = np.array([y for _, y in self.dataset], dtype=np.float32)

        # Get predictions
        print("Generating predictions...")
        batched_dataset = self.dataset.batch(1)
        self.predictions = self.bbox_model.predict(batched_dataset)

        print(f"Predictions shape: {self.predictions.shape}")
        print(f"Test images: {len(self.test_images)}")
        print(f"Ground truth shape: {self.y_true.shape}")

    def visualize(self, slice_idx, obj_thresh=0.5, nms_thresh=0.5, show_confidence=True):
        """Visualize predictions for a specific image index."""
        if slice_idx >= len(self.test_images):
            return plt.figure()

        fig = draw_bounding_boxes(
            self.test_images[slice_idx][..., 0],  # Use first channel for visualization
            self.y_true[slice_idx],
            self.predictions[slice_idx],
            obj_thresh=obj_thresh,
            nms_thresh=nms_thresh,
            show_confidence=show_confidence,
        )
        return fig

    def launch_interface(self):
        """Launch Gradio interface for interactive visualization."""
        interface = gr.Interface(
            fn=self.visualize,
            inputs=[
                gr.Slider(0, len(self.test_images) - 1, step=1, label="Select Image", value=0),
                gr.Slider(0.1, 1.0, step=0.1, label="Objectness Threshold", value=0.5),
                gr.Slider(0.1, 0.9, step=0.1, label="NMS IoU Threshold", value=0.5),
                gr.Checkbox(label="Show Confidence Scores", value=True),
            ],
            outputs=gr.Plot(),
            title="YOLO Methane Plume Detection Visualizer",
            description="Visualize YOLO model predictions for methane plume detection. Green boxes are ground truth, red boxes are predictions.",
            live=True,
        )
        interface.launch()


def main():
    parser = argparse.ArgumentParser(description="Visualize YOLO model predictions")
    parser.add_argument("--model-path", type=str, help="Path to model attributes file")
    parser.add_argument(
        "--data-dir", type=str, default="./data/raw_data/STARCOP_test", help="Path to test data directory"
    )
    parser.add_argument("--max-samples", type=int, default=50, help="Maximum number of test samples to load")
    parser.add_argument("--no-gui", action="store_true", help="Run without GUI (for testing)")

    args = parser.parse_args()

    try:
        visualizer = YOLOVisualizer(model_path=args.model_path, data_dir=args.data_dir, max_samples=args.max_samples)

        if args.no_gui:
            # Just test the first image
            fig = visualizer.visualize(0)
            plt.savefig("test_visualization.png")
            plt.close(fig)
            print("Test visualization saved as test_visualization.png")
        else:
            visualizer.launch_interface()

    except Exception as e:
        print(f"Error: {e}")
        # print stack trace for debugging
        import traceback

        traceback.print_exc()
        print("Make sure you have trained a bbox_model_2 model first!")


if __name__ == "__main__":
    main()
