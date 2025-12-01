# functions used to calculate the speed
import numpy as np
from collections import deque

class SpeedTracker:
    def __init__(self, buffer_size=10):
        self.trackers = {}
        self.buffer_size = buffer_size

    def calculate_speed(self, prev_pos, curr_pos, fps):
        """Calculate speed in km/h given two positions and fps."""
        distance = np.sqrt((curr_pos[0] - prev_pos[0])**2 + (curr_pos[1] - prev_pos[1])**2)
        # Calculate time difference based on fps (1/fps = seconds per frame)
        time_diff = 1 / fps
        speed_ms = distance / time_diff if fps > 0 else 0
        return speed_ms * 3.6  # Convert m/s to km/h

    def update_speed(self, track_id, world_coord, frame_count, fps):
        if track_id not in self.trackers:
            self.trackers[track_id] = deque(maxlen=self.buffer_size)
        
        speed_tracker = self.trackers[track_id]
        if speed_tracker:
            speed = self.calculate_speed(speed_tracker[-1][0], world_coord, fps)
        else:
            speed = 0
        
        # Store both position and frame number
        speed_tracker.append((world_coord, frame_count))
        
        # Calculate average speed over the buffer
        if len(speed_tracker) > 1:
            speeds = []
            for i in range(len(speed_tracker) - 1):
                pos1, frame1 = speed_tracker[i]
                pos2, frame2 = speed_tracker[i + 1]
                # Use actual frame difference for more accurate speed calculation
                frame_diff = frame2 - frame1
                if frame_diff > 0:  # Avoid division by zero
                    distance = np.sqrt((pos2[0] - pos1[0])**2 + (pos2[1] - pos1[1])**2)
                    time_diff = frame_diff / fps
                    speed_ms = distance / time_diff
                    speeds.append(speed_ms * 3.6)  # Convert to km/h
            return np.mean(speeds) if speeds else 0
        return speed

    def get_speeds(self, track_ids, world_coords, frame_count, fps):
        return [self.update_speed(track_id, world_coord, frame_count, fps) 
                for track_id, world_coord in zip(track_ids, world_coords)]