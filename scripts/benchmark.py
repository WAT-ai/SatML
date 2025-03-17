import tensorflow as tf
import numpy as np
import pandas as pd
import time
import psutil
import os
import csv
from datetime import datetime
from memory_profiler import profile
import logging
from typing import Tuple, Dict
from sklearn.metrics import f1_score, precision_score, recall_score
from pathlib import Path


class StarcopMethaneDetector:
    def __init__(self, model_path: str):
        """
        Initialize the STARCOP methane detection benchmark.

        Args:
            model_path (str): Path to the TensorFlow model file
        """
        self.model_path = model_path
        self.model_name = os.path.basename(model_path)
        self.results_file = f"starcop_methane_benchmark_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"

        # Setup logging
        logging.basicConfig(level=logging.INFO)
        self.logger = logging.getLogger(__name__)

        # Load model
        try:
            self.model = tf.keras.models.load_model(model_path)
            self.logger.info(f"Successfully loaded model from {model_path}")
        except Exception as e:
            self.logger.error(f"Failed to load model: {e}")
            raise

    def load_starcop_data(self, data_path: str, subset: str = "test") -> Tuple[tf.data.Dataset, pd.DataFrame]:
        """
        Load the STARCOP dataset from the specified path.

        Args:
            data_path: Root path to STARCOP dataset
            subset: Which subset to load ('test', 'train_easy', or 'train_remaining')

        Returns:
            tuple: (dataset, metadata_df)
        """
        self.logger.info(f"Loading STARCOP {subset} data from {data_path}")

        # Load metadata CSV
        csv_path = Path(data_path) / f"STARCOP_{subset}.csv"
        metadata_df = pd.read_csv(csv_path)

        def load_hyperspectral_data(file_path: str) -> np.ndarray:
            """Load hyperspectral data from specified file"""
            return np.load(file_path)

        # Create dataset
        data_files = [str(Path(data_path) / f) for f in metadata_df["file_name"]]
        labels = metadata_df["has_plume"].values

        dataset = tf.data.Dataset.from_tensor_slices((data_files, labels))
        dataset = dataset.map(lambda x, y: (tf.py_function(load_hyperspectral_data, [x], tf.float32), y))

        return dataset, metadata_df

    @profile
    def run_inference(self, dataset: tf.data.Dataset) -> Tuple[np.ndarray, np.ndarray, float]:
        """
        Run model inference on STARCOP dataset and measure execution time.
        """
        self.logger.info("Starting methane plume detection inference")
        start_time = time.time()
        predictions = []
        ground_truth = []

        for images, labels in dataset:
            batch_pred = self.model.predict(images, verbose=0)
            predictions.append(batch_pred)
            ground_truth.append(labels)

        inference_time = time.time() - start_time

        return (np.concatenate(predictions), np.concatenate(ground_truth), inference_time)

    def calculate_starcop_metrics(
        self, predictions: np.ndarray, ground_truth: np.ndarray, metadata_df: pd.DataFrame
    ) -> Dict[str, float]:
        """
        Calculate STARCOP-specific evaluation metrics.

        Args:
            predictions: Model predictions
            ground_truth: True labels
            metadata_df: DataFrame with metadata including qplume values

        Returns:
            Dictionary of metrics including F1 score and false positive rates
        """
        # Convert predictions to binary
        pred_binary = predictions > 0.5

        # Calculate metrics
        metrics = {
            "f1_score": f1_score(ground_truth, pred_binary),
            "precision": precision_score(ground_truth, pred_binary),
            "recall": recall_score(ground_truth, pred_binary),
            "false_positive_rate": np.sum((pred_binary == 1) & (ground_truth == 0)) / len(ground_truth),
        }

        # Calculate metrics for different plume sizes
        if "qplume" in metadata_df.columns:
            large_plumes = metadata_df["qplume"] > 1000
            metrics.update(
                {
                    "f1_score_large_plumes": f1_score(ground_truth[large_plumes], pred_binary[large_plumes]),
                    "f1_score_small_plumes": f1_score(ground_truth[~large_plumes], pred_binary[~large_plumes]),
                }
            )

        return metrics

    def run_benchmark(self, data_path: str, batch_size: int = 32) -> None:
        """
        Run the complete STARCOP benchmark suite.

        Args:
            data_path: Path to STARCOP dataset
            batch_size: Batch size for inference
        """
        try:
            # Load and preprocess STARCOP dataset
            dataset, metadata_df = self.load_starcop_data(data_path)
            dataset = self.preprocess_starcop_data(dataset)
            dataset = dataset.batch(batch_size)

            # Run inference
            predictions, ground_truth, inference_time = self.run_inference(dataset)

            # Measure memory
            memory_usage = self.measure_memory_usage()

            # Calculate metrics
            metrics = self.calculate_starcop_metrics(predictions, ground_truth, metadata_df)

            # Store results
            result = {
                "model_name": self.model_name,
                "inference_time": inference_time,
                "memory_usage_mb": memory_usage,
                **metrics,
            }

            # Save results
            self.save_results([result])

        except Exception as e:
            self.logger.error(f"STARCOP benchmark failed: {e}")
            raise

    def measure_memory_usage(self) -> float:
        """Measure current memory usage."""
        process = psutil.Process(os.getpid())
        memory_mb = process.memory_info().rss / 1024 / 1024
        return memory_mb

    def save_results(self, results: list) -> None:
        """Save benchmark results to CSV file."""
        if not results:
            return

        fieldnames = list(results[0].keys())

        with open(self.results_file, "w", newline="") as csvfile:
            writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(results)

        self.logger.info(f"STARCOP benchmark results saved to {self.results_file}")
