import numpy as np
import matplotlib.pyplot as plt
import os
from feat.utils.tracker import KalmanBoxTracker

def generate_sine_wave_path(steps=100):
    path = []
    for t in range(steps):
        x = 50 + t * 5
        y = 200 + 50 * np.sin(t * 0.1)
        w, h = 50, 50
        path.append([x, y, w, h])
    return np.array(path)

def run_simulation():
    ground_truth = generate_sine_wave_path(steps=100)
    tracker = KalmanBoxTracker(ground_truth[0])
    
    predictions = []
    measurements = []
    
    print("Running Ghost Box Simulation...")
    
    for bbox in ground_truth:
        pred_bbox = tracker.predict()
        predictions.append(pred_bbox[0])
        
        noise = np.random.normal(0, 2, 4) 
        measured_bbox = bbox + noise
        measurements.append(measured_bbox)
        
        tracker.update(measured_bbox)

    measurements = np.array(measurements)
    predictions = np.array(predictions)
    
    plt.figure(figsize=(10, 6))
    plt.plot(measurements[:, 0], measurements[:, 1], 'g.', label='Measurements', alpha=0.5)
    plt.plot(predictions[:, 0], predictions[:, 1], 'b-', label='Kalman Prediction', linewidth=2)
    plt.plot(ground_truth[:, 0], ground_truth[:, 1], 'k--', label='Ground Truth', alpha=0.5)
    plt.legend()
    plt.title("Sprint 2.2 Verification: Kalman Tracker")
    plt.grid(True)
    
    output_file = "tracker_verification.png"
    plt.savefig(output_file)
    print(f"Verification plot saved to {os.path.abspath(output_file)}")

if __name__ == "__main__":
    run_simulation()