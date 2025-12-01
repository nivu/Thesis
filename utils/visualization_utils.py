import cv2

def draw_annotations(image, boxes, keypoints, track_ids, speeds):
    for box, kps, track_id, speed in zip(boxes, keypoints, track_ids, speeds):
        x, y, w, h = box
        # Draw bounding box
        cv2.rectangle(image, (int(x - w/2), int(y - h/2)), (int(x + w/2), int(y + h/2)), (0, 255, 0), 2)
        
        # Draw track ID and speed
        label = f"ID: {track_id}, Speed: {speed:.1f} km/h"
        cv2.putText(image, label, (int(x - w/2), int(y - h/2 - 10)), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
        
        # Draw keypoints
        for kp in kps:
            kp_x, kp_y, kp_conf = kp
            if kp_conf > 0:
                cv2.circle(image, (int(kp_x), int(kp_y)), 5, (255, 0, 0), -1)
    
    return image