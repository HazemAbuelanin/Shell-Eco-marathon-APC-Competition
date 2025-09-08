# Required imports
import numpy as np
from scipy.optimize import linear_sum_assignment

ASSOC_THRESHOLD = 4.5
MAX_MISSES = 5

# Velocity regression
def regression_velocity(history):
    if len(history) < 3:
        return None
    times = np.array([entry[0] for entry in history])
    xs = np.array([entry[1][0] for entry in history])
    ys = np.array([entry[1][1] for entry in history])
    vx = np.polyfit(times, xs, 1)[0]
    vy = np.polyfit(times, ys, 1)[0]
    return np.sqrt(vx**2 + vy**2)



# Kalman Filter class
class KalmanFilter2D:
    def __init__(self, init_pos, init_time):
        self.x = np.array([init_pos[0], init_pos[1], 0.0, 0.0])
        self.P = np.eye(4) * 1.0
        self.last_time = init_time
        self.R = np.eye(2) * 0.1     # Less confidence in measurements
        self.Q = np.diag([0.1, 0.1, 1.0, 1.0])  # Higher velocity noise

    def predict(self, dt):
        F = np.array([[1, 0, dt, 0],
                      [0, 1, 0, dt],
                      [0, 0, 1,  0],
                      [0, 0, 0,  1]])
        self.x = F @ self.x
        self.P = F @ self.P @ F.T + self.Q

    def update(self, measurement):
        H = np.array([[1, 0, 0, 0],
                      [0, 1, 0, 0]])
        y = measurement - H @ self.x
        S = H @ self.P @ H.T + self.R
        K = self.P @ H.T @ np.linalg.inv(S)
        self.x = self.x + K @ y
        self.P = (np.eye(4) - K @ H) @ self.P

    def process(self, measurement, current_time):
        dt = current_time - self.last_time
        if dt < 0.01:
            dt = 0.01
        self.predict(dt)
        self.update(measurement)
        self.last_time = current_time

    def get_velocity(self):
        return self.x[2:4]

    def get_position(self):
        return self.x[0:2]

# Hungarian algorithm-based data association
def match_detections_to_tracks(detections, tracks, max_cost=ASSOC_THRESHOLD):
    if len(detections) == 0 or len(tracks) == 0:
        return [], list(range(len(tracks))), list(range(len(detections)))

    cost_matrix = np.zeros((len(tracks), len(detections)))
    for i, track in enumerate(tracks):
        predicted_pos = track['kf'].get_position()
        for j, det in enumerate(detections):
            cost_matrix[i, j] = np.linalg.norm(predicted_pos - det)

    row_idx, col_idx = linear_sum_assignment(cost_matrix)
    matches = []
    unmatched_tracks = set(range(len(tracks)))
    unmatched_detections = set(range(len(detections)))

    for r, c in zip(row_idx, col_idx):
        if cost_matrix[r, c] < max_cost:
            matches.append((r, c))
            unmatched_tracks.discard(r)
            unmatched_detections.discard(c)

    return matches, list(unmatched_tracks), list(unmatched_detections)

# Main tracking update step
def update_tracking(detections, tracked_objects, current_time, max_misses=5):
    if len(tracked_objects) == 0:
        # Initialize tracks if no existing objects
        new_tracked = {}
        for i, det in enumerate(detections):
            kf = KalmanFilter2D(np.array(det), current_time)
            new_tracked[i] = {
                'kf': kf,
                'history': [(current_time, np.array(det))],
                'misses': 0,
                'last_global': np.array(det),
                'speed': 0.0
            }
        return new_tracked

    # Prepare data
    track_ids = list(tracked_objects.keys())
    tracks_list = [tracked_objects[tid] for tid in track_ids]
    detections_np = [np.array(d) for d in detections]

    # Match detections to existing tracks
    matches, unmatched_tracks, unmatched_detections = match_detections_to_tracks(detections_np, tracks_list)

    new_tracked = {}

    # Update matched tracks
    for track_idx, det_idx in matches:
        obj_id = track_ids[track_idx]
        det = detections_np[det_idx]
        track = tracked_objects[obj_id]
        kf = track['kf']
        kf.process(det, current_time)  # ✅ FIXED: correct Kalman update
        history = track['history'] + [(current_time, det)]
        velocity_kf = np.linalg.norm(kf.get_velocity())

        speed = velocity_kf 

        new_tracked[obj_id] = {
            'kf': kf,
            'history': history,
            'misses': 0,
            'last_global': det,
            'speed': speed
        }

    # Propagate unmatched tracks
    for track_idx in unmatched_tracks:
        obj_id = track_ids[track_idx]
        track = tracked_objects[obj_id]
        kf = track['kf']

        # ✅ FIXED: correctly compute dt and call predict()
        dt = current_time - kf.last_time
        if dt < 0.01:
            dt = 0.01
        kf.predict(dt)
        kf.last_time = current_time

        new_pos = kf.get_position()
        if track['misses'] <= max_misses:
            new_tracked[obj_id] = {
                'kf': kf,
                'history': track['history'],
                'misses': track['misses'] + 1,
                'last_global': new_pos,
                'speed': 0.0
            }

    # Create new tracks for unmatched detections
    next_id = max(new_tracked.keys(), default=-1) + 1
    for idx in unmatched_detections:
        det = detections_np[idx]
        kf = KalmanFilter2D(det, current_time)
        new_tracked[next_id] = {
            'kf': kf,
            'history': [(current_time, det)],
            'misses': 0,
            'last_global': det,
            'speed': 0.0
        }
        next_id += 1

    # Remove tracks that have missed too many frames
    tracked_objects = {tid: data for tid, data in new_tracked.items() if data['misses'] <= max_misses}


    return tracked_objects
# Instantaneous velocity calculation
# This function calculates the instantaneous velocity based on the last two positions in the history.
# It returns 0.0 if the time difference is too small or if there are not enough points in the history.
def instant_velocity(history):
    if len(history) < 2:
        return 0.0
    t1, p1 = history[-2]
    t2, p2 = history[-1]
    dt = t2 - t1
    if dt <= 0.01:
        return 0.0
    return np.linalg.norm(p2 - p1) / dt