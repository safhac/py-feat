import numpy as np
from feat.utils.tracker import KalmanBoxTracker


def test_kalman_tracker_lifecycle():
    # 1. Initialize
    initial_bbox = [100, 100, 50, 50]  # x, y, w, h
    tracker = KalmanBoxTracker(initial_bbox)

    # 2. Test Predict (Should move slightly or stay depending on velocity init)
    prediction = tracker.predict()
    assert prediction.shape == (1, 4), "Prediction should return 1x4 bbox"

    # 3. Test Update (Simulate face moving to right)
    new_bbox = [105, 100, 50, 50]
    tracker.update(new_bbox)

    # 4. Check State
    current_state = tracker.get_state()
    assert current_state.shape == (1, 4), "Get state should return 1x4 bbox"

    # 5. Check Consistency
    # The tracker x position should be somewhere between 100 and 105 (smoothing)
    # or close to 105.
    pred_x = current_state[0, 0]
    assert 90 < pred_x < 115, f"Tracker prediction {pred_x} is way off!"
