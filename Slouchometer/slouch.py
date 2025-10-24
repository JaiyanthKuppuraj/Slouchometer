# slouchometer_step3_with_logging_v2.py
import cv2
import mediapipe as mp
import math
import time
import csv
import os

mp_pose = mp.solutions.pose
pose = mp_pose.Pose(min_detection_confidence=0.5, min_tracking_confidence=0.5)
mp_drawing = mp.solutions.drawing_utils

# Adjustable parameters
FORWARD_ANGLE_THRESHOLD_DEFAULT = 18.0  # degrees
CALIBRATION_OFFSET = 10.0               # degrees above baseline
SHOW_DEBUG_VALUES = True

# CSV log file
LOG_FILE = "posture_log.csv"

# If file doesn't exist, create it with header
if not os.path.exists(LOG_FILE):
    with open(LOG_FILE, mode="w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["timestamp", "angle", "status"])

def log_posture(angle, status):
    """Append posture event to CSV log file"""
    with open(LOG_FILE, mode="a", newline="") as f:
        writer = csv.writer(f)
        writer.writerow([time.strftime("%Y-%m-%d %H:%M:%S"), f"{angle:.2f}", status])
    print(f"Logged: {status} at {time.strftime('%H:%M:%S')} angle={angle:.2f}")

def get_lm(landmarks, lm):
    lm_obj = landmarks[lm]
    return {'x': lm_obj.x, 'y': lm_obj.y, 'z': lm_obj.z, 'vis': getattr(lm_obj, 'visibility', 1.0)}

def forward_neck_angle(shoulder, ear):
    vy = ear['y'] - shoulder['y']
    vz = shoulder['z'] - ear['z']
    vy = vy if abs(vy) > 1e-6 else 1e-6
    angle_rad = math.atan2(abs(vz), abs(vy))
    return math.degrees(angle_rad)

def to_pixel_coords(norm_x, norm_y, img_w, img_h):
    return (int(norm_x * img_w), int(norm_y * img_h))

cap = cv2.VideoCapture(0)
calibrated_baseline = None
threshold = FORWARD_ANGLE_THRESHOLD_DEFAULT
last_cal_time = 0
last_status = None  # track previous posture status

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    h, w, _ = frame.shape
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = pose.process(rgb)

    status_text = "No person detected"
    left_angle = right_angle = None
    angle_used = None
    posture_status = None

    if results.pose_landmarks:
        lm = results.pose_landmarks.landmark

        left_sh = get_lm(lm, mp_pose.PoseLandmark.LEFT_SHOULDER)
        right_sh = get_lm(lm, mp_pose.PoseLandmark.RIGHT_SHOULDER)
        left_ear = get_lm(lm, mp_pose.PoseLandmark.LEFT_EAR)
        right_ear = get_lm(lm, mp_pose.PoseLandmark.RIGHT_EAR)
        nose = get_lm(lm, mp_pose.PoseLandmark.NOSE)

        if left_ear['vis'] < 0.4:
            left_ear = nose
        if right_ear['vis'] < 0.4:
            right_ear = nose

        try:
            left_angle = forward_neck_angle(left_sh, left_ear)
        except Exception:
            left_angle = None
        try:
            right_angle = forward_neck_angle(right_sh, right_ear)
        except Exception:
            right_angle = None

        angles_available = [a for a in (left_angle, right_angle) if a is not None]
        if angles_available:
            angle_used = max(angles_available)

        if angle_used is not None:
            if calibrated_baseline is not None:
                thr = calibrated_baseline + CALIBRATION_OFFSET
            else:
                thr = threshold

            if angle_used > thr:
                status_text = f"Bad Posture (forward angle {angle_used:.1f}° > thr {thr:.1f}°)"
                color = (0, 0, 255)
                posture_status = "Bad"
            else:
                status_text = f"Good Posture (angle {angle_used:.1f}°)"
                color = (0, 255, 0)
                posture_status = "Good"
        else:
            status_text = "Could not compute angles"
            color = (0, 255, 255)

        for side, sh, ear_point, angle in [
            ('L', left_sh, left_ear, left_angle),
            ('R', right_sh, right_ear, right_angle),
        ]:
            px_sh = to_pixel_coords(sh['x'], sh['y'], w, h)
            px_ear = to_pixel_coords(ear_point['x'], ear_point['y'], w, h)
            cv2.circle(frame, px_sh, 6, (255, 165, 0), -1)
            cv2.circle(frame, px_ear, 6, (255, 165, 0), -1)
            cv2.line(frame, px_ear, px_sh, (255, 165, 0), 2)
            if SHOW_DEBUG_VALUES and angle is not None:
                cv2.putText(frame, f'{side}:{angle:.1f}°', (px_sh[0]+8, px_sh[1]-8),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200,200,200), 1)

        if calibrated_baseline is not None:
            cv2.putText(frame, f'Baseline: {calibrated_baseline:.1f} deg (thr={calibrated_baseline+CALIBRATION_OFFSET:.1f})',
                        (10, h-20), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (200,200,200), 2)
        else:
            cv2.putText(frame, f'Default thr: {threshold:.1f} deg — press c to calibrate neutral', (10, h-20),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200,200,200), 1)

        cv2.putText(frame, status_text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2)

        # ✅ Log only when posture changes (Good <-> Bad)
        if posture_status in ("Good", "Bad") and posture_status != last_status:
            log_posture(angle_used, posture_status)
            last_status = posture_status

    else:
        cv2.putText(frame, "No person detected", (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0,255,255), 2)

    cv2.imshow('Slouchometer - Step 3 (logging transitions)', frame)
    key = cv2.waitKey(1) & 0xFF

    if key == ord('c'):
        if left_angle is not None and right_angle is not None:
            calibrated_baseline = (left_angle + right_angle) / 2.0
        elif left_angle is not None:
            calibrated_baseline = left_angle
        elif right_angle is not None:
            calibrated_baseline = right_angle
        else:
            calibrated_baseline = None
        last_cal_time = time.time()
        print(f"Calibrated baseline: {calibrated_baseline}")
    elif key == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
