import time
import torch
import numpy as np
import pandas as pd
from unittest.mock import patch, MagicMock
from feat.detector import Detector


# A simple mock dataset
class MockVideoDataset:
    def __init__(self, length=50):
        self.length = length

    def __len__(self):
        return self.length

    def __getitem__(self, idx):
        if idx >= self.length:
            raise IndexError
        return {
            "Image": torch.randn(1, 3, 112, 112),
            "Frame": torch.tensor([idx]),
            "FileName": [f"frame_{idx}.jpg"],
            "Padding": {
                "Left": torch.tensor([0]),
                "Top": torch.tensor([0]),
                "Right": torch.tensor([0]),
                "Bottom": torch.tensor([0]),
            },
            "Scale": torch.tensor([1.0]),
        }

    def calc_approx_frame_time(self, frame_idx):
        return frame_idx / 30.0


def run_benchmark():
    print("\n=== SPRINT 4.2: SIMULATED LOAD TEST ===")

    # 1. Setup
    detector = Detector(
        landmark_model="mobilefacenet",
        au_model=None,
        emotion_model=None,
        identity_model=None,
    )

    N_FRAMES = 50

    # --- THE SIMULATION MOCK ---
    # This mocks the heavy 'detect_faces' (img2pose) method.
    # It sleeps for 0.1s to simulate GPU work and returns a dummy face.
    def mock_heavy_detect(images, **kwargs):
        time.sleep(0.1)  # <--- Simulate Heavy GPU Load

        # Return a fixed face box [100, 100, 200, 200]
        box = torch.tensor([[100.0, 100.0, 200.0, 200.0]])
        return [
            {
                "boxes": box,
                "scores": torch.tensor([0.99]),
                "faces": torch.zeros((1, 3, 112, 112)),
                "poses": torch.zeros((1, 6)),
                "new_boxes": box,
            }
        ]

    # Apply the mock
    detector.detect_faces = mock_heavy_detect

    print(f"Benchmarking on {N_FRAMES} frames.")
    print("Simulated Load per Detection: 0.1s")

    with (
        patch(
            "feat.detector.VideoDataset", return_value=MockVideoDataset(length=N_FRAMES)
        ),
        patch("feat.detector.DataLoader", side_effect=lambda x, **kwargs: x),
    ):
        # -------------------------------------------------
        # RUN 1: BASELINE (Interval=1) -> Hits the 0.1s sleep EVERY frame
        # -------------------------------------------------
        print("\n1. Running BASELINE (Interval=1)...")
        start_time = time.time()
        detector.detect(
            inputs="dummy.mp4",
            data_type="video",
            batch_size=1,
            detection_interval=1,
            progress_bar=True,
        )
        baseline_time = time.time() - start_time
        baseline_fps = N_FRAMES / baseline_time
        print(f"   -> Time: {baseline_time:.2f}s | FPS: {baseline_fps:.2f}")

        # -------------------------------------------------
        # RUN 2: HYBRID (Interval=5) -> Hits the 0.1s sleep only 20% of the time
        # -------------------------------------------------
        print("\n2. Running HYBRID (Interval=5)...")
        # Reset trackers for fairness
        detector.trackers = []

        start_time = time.time()
        detector.detect(
            inputs="dummy.mp4",
            data_type="video",
            batch_size=1,
            detection_interval=5,
            progress_bar=True,
        )
        hybrid_time = time.time() - start_time
        hybrid_fps = N_FRAMES / hybrid_time
        print(f"   -> Time: {hybrid_time:.2f}s | FPS: {hybrid_fps:.2f}")

        # -------------------------------------------------
        # RESULTS
        # -------------------------------------------------
        speedup = (baseline_time - hybrid_time) / baseline_time * 100
        factor = hybrid_fps / baseline_fps
        print(f"\n=============================================")
        print(f"SPEEDUP ACHIEVED: {speedup:.1f}%")
        print(f"PERFORMANCE BOOST: {factor:.1f}x Faster")
        print(f"=============================================")


if __name__ == "__main__":
    run_benchmark()
