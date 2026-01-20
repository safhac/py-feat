import unittest
from unittest.mock import MagicMock, patch
import torch
import numpy as np
import pandas as pd
from feat.detector import Detector


class TestTrackingIntegration(unittest.TestCase):
    def setUp(self):
        # Initialize detector with minimal config
        self.detector = Detector(
            landmark_model=None, au_model=None, emotion_model=None, identity_model=None
        )

    def _get_mock_loader(self, num_frames=10):
        mock_loader = []
        for i in range(num_frames):
            mock_loader.append(
                {
                    "Image": torch.randn(1, 3, 112, 112),
                    "Frame": torch.tensor([i]),
                    "FileName": [f"frame_{i}.jpg"],
                    "Padding": {
                        "Left": torch.tensor([0]),
                        "Top": torch.tensor([0]),
                        "Right": torch.tensor([0]),
                        "Bottom": torch.tensor([0]),
                    },
                    "Scale": torch.tensor([1.0]),
                }
            )
        return mock_loader

    def _mock_detect_faces_output(self, images, **kwargs):
        # Determine current frame from internal counter
        frame_idx = self.detector.test_frame_counter
        self.detector.test_frame_counter += 1

        # Create a moving box [x1, y1, x2, y2]
        shift = frame_idx * 5
        box = torch.tensor([[100.0 + shift, 100.0 + shift, 200.0 + shift, 200.0 + shift]])

        return [
            {
                "boxes": box,
                "scores": torch.tensor([0.99]),
                "faces": torch.zeros((1, 3, 112, 112)),
                "poses": torch.zeros((1, 6)),
                "new_boxes": box,
            }
        ]

    def test_tracking_logic_is_active(self):
        """
        Sprint 3.3 Test: Verify TrackIDs are assigned.
        """
        print("\n--- Test 1: Tracking Logic Integration ---")
        with (
            patch("feat.detector.VideoDataset"),
            patch("feat.detector.DataLoader", return_value=self._get_mock_loader(5)),
        ):
            self.detector.test_frame_counter = 0
            self.detector.detect_faces = self._mock_detect_faces_output

            prediction = self.detector.detect(
                inputs="dummy.mp4", data_type="video", batch_size=1, detection_interval=1
            )

            self.assertIn("TrackID", prediction.columns)
            unique_ids = prediction["TrackID"].unique()
            print(f"Unique IDs: {unique_ids}")
            self.assertEqual(len(unique_ids), 1)

    def test_hybrid_switching_skips_detection(self):
        """
        Sprint 3.2 Test: Verify that detection_interval > 1 reduces calls to detect_faces.
        """
        print("\n--- Test 2: Hybrid Switch Efficiency ---")
        num_frames = 10
        interval = 5

        with (
            patch("feat.detector.VideoDataset"),
            patch(
                "feat.detector.DataLoader", return_value=self._get_mock_loader(num_frames)
            ),
        ):
            # Setup Spy: We wrap the mock so we can count calls
            self.detector.test_frame_counter = 0
            # Create a spy that has the side effect of our mock output
            spy = MagicMock(side_effect=self._mock_detect_faces_output)
            self.detector.detect_faces = spy

            # RUN with interval=5
            prediction = self.detector.detect(
                inputs="dummy.mp4",
                data_type="video",
                batch_size=1,
                detection_interval=interval,
            )

            # CHECK RESULTS
            # Expected calls: Frame 0 (Init), Frame 5 (Correction).
            # Frames 1,2,3,4,6,7,8,9 should be skipped.
            # Total expected: 2 calls.

            print(f"Total Frames: {num_frames}")
            print(f"Detection Interval: {interval}")
            print(f"Actual Detector Calls: {spy.call_count}")

            # We assert we saved at least 50% of calls
            self.assertLess(
                spy.call_count,
                num_frames / 2,
                "Hybrid Switch failed! Detector was called too many times.",
            )
            self.assertEqual(
                spy.call_count,
                2,
                "Exact call count mismatch. Expected detection on frame 0 and 5.",
            )

            # Verify we still got results for ALL frames
            self.assertEqual(
                len(prediction), num_frames, "We lost frames! Output length mismatch."
            )
            print("SUCCESS: Hybrid Logic skipped frames correctly.")


if __name__ == "__main__":
    unittest.main()
