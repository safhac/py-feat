import unittest
from unittest.mock import MagicMock, patch
import torch
import numpy as np
import pandas as pd
from feat.detector import Detector


class TestTrackingIntegration(unittest.TestCase):
    def setUp(self):
        # Initialize detector with minimal config to save memory
        self.detector = Detector(
            landmark_model=None, au_model=None, emotion_model=None, identity_model=None
        )

    def test_tracking_logic_is_active(self):
        """
        Simulate a video with a moving face and verify TrackIDs are assigned and consistent.
        """
        print("\n--- Starting Tracking Integration Test ---")

        # 1. Mock the VideoDataset so we don't need a real video file
        # It needs to return a batch dictionary like the real DataLoader
        with patch("feat.detector.VideoDataset") as MockDataset:
            # Create a dummy loader that yields 5 frames
            mock_loader = []
            for i in range(5):
                mock_loader.append(
                    {
                        "Image": torch.randn(1, 3, 112, 112),
                        "Frame": torch.tensor([i]),
                        "FileName": [f"frame_{i}.jpg"],
                        # UPDATE: Added 'Right' and 'Bottom' to prevent KeyError
                        "Padding": {
                            "Left": torch.tensor([0]),
                            "Top": torch.tensor([0]),
                            "Right": torch.tensor([0]),
                            "Bottom": torch.tensor([0]),
                        },
                        "Scale": torch.tensor([1.0]),
                    }
                )

            # Patch DataLoader to return our mock_loader
            with patch("feat.detector.DataLoader", return_value=mock_loader):
                # 2. Mock detect_faces so we don't need the AI model
                # We simulate one face moving diagonally: (100,100) -> (105,105) -> ...
                def mock_detect_faces_output(images, **kwargs):
                    results = []
                    # We know the loader batch size is 1 for this test
                    frame_idx = self.detector.test_frame_counter

                    # Create a moving box [x1, y1, x2, y2]
                    shift = frame_idx * 5
                    box = torch.tensor(
                        [[100.0 + shift, 100.0 + shift, 200.0 + shift, 200.0 + shift]]
                    )

                    results.append(
                        {
                            "boxes": box,
                            "scores": torch.tensor([0.99]),
                            "faces": torch.zeros((1, 3, 112, 112)),
                            "poses": torch.zeros((1, 6)),
                            "new_boxes": box,  # For this test, new_boxes can match boxes
                        }
                    )
                    self.detector.test_frame_counter += 1
                    return results

                # Attach the mock to the detector instance
                self.detector.test_frame_counter = 0
                self.detector.detect_faces = mock_detect_faces_output

                # 3. RUN THE DETECT METHOD
                # We pass data_type="video" to trigger your new logic
                prediction = self.detector.detect(
                    inputs="dummy_path.mp4", data_type="video", batch_size=1
                )

                # 4. VERIFY RESULTS
                print("Columns found:", prediction.columns.tolist())

                # Check A: Does TrackID exist?
                self.assertIn(
                    "TrackID",
                    prediction.columns,
                    "TrackID column is missing! The logic is not triggering.",
                )

                # Check B: Are TrackIDs integers?
                self.assertTrue(
                    pd.api.types.is_integer_dtype(prediction["TrackID"]),
                    "TrackID should be integers.",
                )

                # Check C: Consistency
                # Since it's the same face moving smoothly, the ID should stay the same (e.g., 0)
                unique_ids = prediction["TrackID"].unique()
                print(f"Unique Track IDs found: {unique_ids}")
                self.assertEqual(
                    len(unique_ids),
                    1,
                    "The tracker failed to hold the ID for a simple moving face.",
                )
                self.assertNotEqual(
                    unique_ids[0],
                    -1,
                    "The TrackID returned -1, meaning no association was made.",
                )

                print("SUCCESS: System Integration Verified.")


if __name__ == "__main__":
    unittest.main()
