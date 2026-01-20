import numpy as np
from filterpy.kalman import KalmanFilter


class KalmanBoxTracker(object):
    """
    This class represents the internal state of individual tracked objects observed as bbox.
    """

    count = 0

    def __init__(self, bbox):
        """
        Initializes a tracker using initial bounding box.
        bbox must be [x, y, width, height]
        """
        self.kf = KalmanFilter(dim_x=7, dim_z=4)

        # State vector: [x, y, area, ratio, v_x, v_y, v_area]
        self.kf.F = np.array(
            [
                [1, 0, 0, 0, 1, 0, 0],
                [0, 1, 0, 0, 0, 1, 0],
                [0, 0, 1, 0, 0, 0, 1],
                [0, 0, 0, 1, 0, 0, 0],
                [0, 0, 0, 0, 1, 0, 0],
                [0, 0, 0, 0, 0, 1, 0],
                [0, 0, 0, 0, 0, 0, 1],
            ]
        )

        self.kf.H = np.array(
            [
                [1, 0, 0, 0, 0, 0, 0],
                [0, 1, 0, 0, 0, 0, 0],
                [0, 0, 1, 0, 0, 0, 0],
                [0, 0, 0, 1, 0, 0, 0],
            ]
        )

        self.kf.R[2:, 2:] *= 10.0  # Measurement noise
        self.kf.P[4:, 4:] *= 1000.0  # High uncertainty for initial velocities
        self.kf.P *= 10.0
        self.kf.Q[-1, -1] *= 0.01
        self.kf.Q[4:, 4:] *= 0.01

        # Convert [x,y,w,h] to [x, y, area, ratio]
        self.kf.x[:4] = self.convert_bbox_to_z(bbox)

        self.time_since_update = 0
        self.id = KalmanBoxTracker.count
        KalmanBoxTracker.count += 1
        self.history = []
        self.hits = 0
        self.hit_streak = 0
        self.age = 0

    def update(self, bbox):
        """
        Updates the state vector with observed bbox.
        """
        self.time_since_update = 0
        self.history = []
        self.hits += 1
        self.hit_streak += 1
        self.kf.update(self.convert_bbox_to_z(bbox))

    def predict(self):
        """
        Advances the state vector and returns the predicted bounding box estimate.
        """
        if (self.kf.x[6] + self.kf.x[2]) <= 0:
            self.kf.x[6] *= 0.0

        self.kf.predict()
        self.age += 1
        if self.time_since_update > 0:
            self.hit_streak = 0
        self.time_since_update += 1
        self.history.append(self.convert_x_to_bbox(self.kf.x))
        return self.history[-1]

    def get_state(self):
        """
        Returns the current bounding box estimate.
        """
        return self.convert_x_to_bbox(self.kf.x)

    def convert_bbox_to_z(self, bbox):
        """
        Takes a bounding box in the form [x, y, w, h] and returns z in the form
        [x, y, s, r] where x,y is the centre of the box and s is the scale/area and r is
        the aspect ratio
        """
        w = bbox[2]
        h = bbox[3]
        x = bbox[0] + w / 2.0
        y = bbox[1] + h / 2.0
        s = w * h  # scale is just area
        r = w / float(h)
        return np.array([x, y, s, r]).reshape((4, 1))

    def convert_x_to_bbox(self, x, score=None):
        """
        Takes a bounding box in the centre form [x,y,s,r] and returns it in the form
        [x, y, w, h]
        """
        w = np.sqrt(x[2] * x[3])
        h = x[2] / w
        if score == None:
            return np.array([x[0] - w / 2.0, x[1] - h / 2.0, w, h]).reshape((1, 4))
        else:
            return np.array([x[0] - w / 2.0, x[1] - h / 2.0, w, h, score]).reshape((1, 5))
