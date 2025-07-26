import cv2
import numpy as np
import mediapipe as mp
import logging
from collections import deque
import time
import uuid
import os

# Disable TensorFlow oneDNN optimizations to suppress warning
os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# Set FFmpeg environment variables for RTSP (force TCP transport)
os.environ["OPENCV_FFMPEG_CAPTURE_OPTIONS"] = "rtsp_transport;tcp|buffer_size;10485760"

# MediaPipe setup
mp_face_mesh = mp.solutions.face_mesh
mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles

# Liveness detection parameters (relaxed for robustness)
EYE_AR_THRESH = 0.1  # Relaxed for blinks
EYE_AR_CONSEC_FRAMES = 2  # Fast blink detection
GAZE_CHANGE_THRESH = 0.01  # Relaxed for 3D eye movement
LIP_MOVEMENT_THRESH = 2.0  # Relaxed for 3D lip movement
FACE_TURN_THRESH = 0.03  # Relaxed for 3D face rotation
DEPTH_VARIANCE_THRESH = 0.0005  # Kept low for varied conditions
MIN_LIVE_FRAMES = 5  # Fast detection
BLINK_INTERVAL = 5.5  # Wide interval for natural blinks
TEMPORAL_WINDOW = 10  # Short window for responsiveness
LIGHTING_ADAPT_FACTOR = 0.9  # Aggressive adaptation for lighting
SMOOTHING_WINDOW = 5  # Frames for majority vote
MAX_FAILED_FRAMES = 10  # Max consecutive failed frames before giving up

# Calculate Eye Aspect Ratio (EAR)
def eye_aspect_ratio(eye_pts):
    try:
        A = np.linalg.norm(np.array(eye_pts[1]) - np.array(eye_pts[5]))
        B = np.linalg.norm(np.array(eye_pts[2]) - np.array(eye_pts[4]))
        C = np.linalg.norm(np.array(eye_pts[0]) - np.array(eye_pts[3]))
        return (A + B) / (2.0 * C)
    except (IndexError, TypeError):
        return None

# Calculate 3D gaze direction
def gaze_direction(eye_pts, iris_pts, landmarks, indices):
    try:
        eye_center = np.mean([(p[0], p[1], landmarks[i].z) for i, p in zip(indices[:6], eye_pts)], axis=0)
        iris_center = np.mean([(p[0], p[1], landmarks[i].z) for i, p in zip(indices[6:], iris_pts)], axis=0)
        return np.linalg.norm(iris_center - eye_center)
    except (IndexError, TypeError, AttributeError):
        return None

# Calculate 3D lip movement
def lip_movement(lip_pts, landmarks, indices):
    try:
        upper_lip = np.array([lip_pts[0][0], lip_pts[0][1], landmarks[indices[0]].z])
        lower_lip = np.array([lip_pts[1][0], lip_pts[1][1], landmarks[indices[1]].z])
        return np.linalg.norm(upper_lip - lower_lip)
    except (IndexError, TypeError, AttributeError):
        return None

# Calculate 3D face rotation
def face_rotation(landmarks, indices):
    try:
        nose_tip = landmarks.landmark[indices[0]]
        forehead = landmarks.landmark[indices[1]]
        vector = np.array([nose_tip.x - forehead.x, nose_tip.y - forehead.y, nose_tip.z - forehead.z])
        return np.linalg.norm(vector)
    except (IndexError, TypeError, AttributeError):
        return None

# Calculate depth variance
def depth_variance(landmarks):
    try:
        z_coords = [landmark.z for landmark in landmarks.landmark[:10]]  # Use first 10 landmarks for stability
        return np.var(z_coords)
    except (AttributeError, TypeError):
        return None

# Estimate lighting conditions
def estimate_lighting(frame):
    try:
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        return np.mean(gray) / 255.0  # Normalized brightness (0 to 1)
    except Exception as e:
        logging.error(f"Lighting estimation error: {e}")
        return 0.5  # Fallback value

# Calibrate thresholds
def calibrate_threshold(history, base_threshold, lighting_factor):
    try:
        if len(history) < TEMPORAL_WINDOW:
            return base_threshold
        variance = np.var(list(history)) if history else 0
        return base_threshold * (LIGHTING_ADAPT_FACTOR * lighting_factor + variance / 5.0)
    except Exception as e:
        logging.error(f"Threshold calibration error: {e}")
        return base_threshold

# Ensure square frame for MediaPipe
def ensure_square_frame(frame, target_size):
    try:
        h, w, _ = frame.shape
        size = min(h, w)
        start_x = (w - size) // 2
        start_y = (h - size) // 2
        square_frame = frame[start_y:start_y + size, start_x:start_x + size]
        square_frame = cv2.resize(square_frame, target_size)
        return square_frame
    except Exception as e:
        logging.error(f"Square frame processing error: {e}")
        return None

# Face tracking class
class FaceTracker:
    def __init__(self):
        self.tracked_faces = {}  # {track_id: {center, last_seen, live_frames}}
        self.live_people = set()
        self.MAX_INACTIVE_FRAMES = 5

    def update(self, face_center, is_live, frame_count):
        try:
            track_id = None
            min_dist = float('inf')
            threshold = 200  # Increased for multi-face tracking

            for tid, data in list(self.tracked_faces.items()):
                dist = np.linalg.norm(np.array(face_center) - np.array(data['center']))
                if dist < min_dist and dist < threshold:
                    min_dist = dist
                    track_id = tid

            if track_id is None:
                track_id = str(uuid.uuid4())
                self.tracked_faces[track_id] = {
                    'center': face_center,
                    'last_seen': frame_count,
                    'live_frames': 0
                }
            else:
                self.tracked_faces[track_id]['center'] = face_center
                self.tracked_faces[track_id]['last_seen'] = frame_count
                if is_live:
                    self.tracked_faces[track_id]['live_frames'] += 1
                    if self.tracked_faces[track_id]['live_frames'] >= MIN_LIVE_FRAMES:
                        self.live_people.add(track_id)

            for tid in list(self.tracked_faces):
                if frame_count - self.tracked_faces[tid]['last_seen'] > self.MAX_INACTIVE_FRAMES:
                    del self.tracked_faces[tid]
                    self.live_people.discard(tid)
        except Exception as e:
            logging.error(f"FaceTracker update error: {e}")

    def get_live_count(self):
        return len(self.live_people)

# Liveness detector class
class LivenessDetector:
    def __init__(self):
        self.blink_counter = 0
        self.consecutive_frames = 0
        self.last_blink_time = time.time()
        self.gaze_history = deque(maxlen=TEMPORAL_WINDOW)
        self.lip_history = deque(maxlen=TEMPORAL_WINDOW)
        self.rotation_history = deque(maxlen=TEMPORAL_WINDOW)
        self.depth_history = deque(maxlen=TEMPORAL_WINDOW)
        self.prev_ear = None
        self.liveness_confidence = 0.0
        self.live_votes = deque(maxlen=SMOOTHING_WINDOW)  # For majority voting

    def is_live_face(self, landmarks, img_shape, lighting_factor):
        h, w = img_shape[:2]
        try:
            # Landmark indices (minimal for robustness)
            left_eye_indices = [33, 160, 158, 133, 153, 144]
            right_eye_indices = [362, 385, 387, 263, 373, 380]
            iris_indices = [468, 473]  # One iris point per eye
            lip_indices = [0, 17]
            rotation_indices = [1, 10]  # Nose, forehead

            # Validate indices
            max_index = len(landmarks.landmark) - 1
            required_indices = left_eye_indices + right_eye_indices + iris_indices + lip_indices + rotation_indices
            if any(i > max_index for i in required_indices):
                logging.debug("Invalid landmark indices detected")
                return False, 0.0

            # Extract landmarks
            left_eye = [landmarks.landmark[i] for i in left_eye_indices]
            right_eye = [landmarks.landmark[i] for i in right_eye_indices]
            iris = [landmarks.landmark[i] for i in iris_indices]
            lips = [landmarks.landmark[i] for i in lip_indices]

            # Convert to pixel coordinates
            left_eye_pts = [(p.x * w, p.y * h) for p in left_eye]
            right_eye_pts = [(p.x * w, p.y * h) for p in right_eye]
            iris_pts = [(p.x * w, p.y * h) for p in iris]
            lips_pts = [(p.x * w, p.y * h) for p in lips]

            # Blink detection
            ear = eye_aspect_ratio(left_eye_pts + right_eye_pts)
            if ear is None:
                logging.debug("EAR calculation failed")
                return False, 0.0
            if self.prev_ear is not None and abs(ear - self.prev_ear) > 0.2:
                logging.debug("Sudden EAR change detected, likely spoof")
                return False, 0.0
            self.prev_ear = ear

            if ear < EYE_AR_THRESH:
                self.consecutive_frames += 1
            else:
                if self.consecutive_frames >= EYE_AR_CONSEC_FRAMES:
                    self.blink_counter += 1
                    self.last_blink_time = time.time()
                self.consecutive_frames = 0

            # 3D gaze detection
            gaze = gaze_direction(left_eye_pts + right_eye_pts, iris_pts, landmarks.landmark, left_eye_indices + right_eye_indices + iris_indices)
            if gaze is None:
                logging.debug("Gaze calculation failed")
                return False, 0.0
            self.gaze_history.append(gaze)

            # 3D lip movement
            lip_dist = lip_movement(lips_pts, landmarks.landmark, lip_indices)
            if lip_dist is None:
                logging.debug("Lip movement calculation failed")
                return False, 0.0
            self.lip_history.append(lip_dist)

            # 3D face rotation
            rotation = face_rotation(landmarks, rotation_indices)
            if rotation is None:
                logging.debug("Rotation calculation failed")
                return False, 0.0
            self.rotation_history.append(rotation)

            # Depth variance
            depth_var = depth_variance(landmarks)
            if depth_var is None:
                logging.debug("Depth variance calculation failed")
                return False, 0.0
            self.depth_history.append(depth_var)

            # Calibrate thresholds
            gaze_thresh = calibrate_threshold(self.gaze_history, GAZE_CHANGE_THRESH, lighting_factor)
            lip_thresh = calibrate_threshold(self.lip_history, LIP_MOVEMENT_THRESH, lighting_factor)
            rotation_thresh = calibrate_threshold(self.rotation_history, FACE_TURN_THRESH, lighting_factor)

            # Temporal consistency
            gaze_change = max(self.gaze_history) - min(self.gaze_history) if self.gaze_history else 0
            lip_change = max(self.lip_history) - min(self.lip_history) if self.lip_history else 0
            rotation_change = max(self.rotation_history) - min(self.rotation_history) if self.rotation_history else 0
            depth_var = np.mean(self.depth_history) if self.depth_history else 0

            # Liveness criteria
            time_since_blink = time.time() - self.last_blink_time
            is_blinking = self.blink_counter > 0 and time_since_blink < BLINK_INTERVAL
            is_gaze_changing = gaze_change > gaze_thresh
            is_lip_moving = lip_change > lip_thresh
            is_face_turning = rotation_change > rotation_thresh
            is_depth_varied = depth_var > DEPTH_VARIANCE_THRESH

            # Calculate confidence
            confidence = sum([
                0.3 if is_blinking else 0.0,
                0.25 if is_gaze_changing else 0.0,
                0.25 if is_lip_moving else 0.0,
                0.2 if is_face_turning else 0.0,
                0.2 if is_depth_varied else 0.0
            ]) / 1.2  # Normalize to 0.0–1.0
            self.liveness_confidence = confidence

            # Relaxed liveness check with majority voting
            criteria_met = sum([is_blinking, is_gaze_changing, is_lip_moving, is_face_turning, is_depth_varied])
            current_is_live = criteria_met >= 3
            self.live_votes.append(current_is_live)
            is_live = sum(self.live_votes) > len(self.live_votes) // 2  # Majority vote

            logging.debug(f"Live: {is_live}, Confidence: {confidence:.2f}, "
                         f"Blink: {is_blinking} (EAR: {ear:.2f}/{EYE_AR_THRESH}), "
                         f"Gaze: {is_gaze_changing} ({gaze_change:.3f}/{gaze_thresh:.3f}), "
                         f"Lip: {is_lip_moving} ({lip_change:.1f}/{lip_thresh:.1f}), "
                         f"Rotation: {is_face_turning} ({rotation_change:.2f}/{rotation_thresh:.2f}), "
                         f"Depth: {is_depth_varied} ({depth_var:.4f}/{DEPTH_VARIANCE_THRESH}), "
                         f"Criteria Met: {criteria_met}/5, Votes: {sum(self.live_votes)}/{len(self.live_votes)}")
            return is_live, confidence

        except Exception as e:
            logging.error(f"Liveness detection error: {e}")
            return False, 0.0

# Initialize
tracker = FaceTracker()
liveness_detectors = {}
rtsp_url = "rtsp://admin:admin123@192.168.1.108:554/cam/realmonitor?channel=1&subtype=0"
max_retries = 5  # Increased retries
retry_delay = 10  # Increased delay (seconds)

def connect_camera(url, retries, delay):
    for attempt in range(retries):
        try:
            cap = cv2.VideoCapture(url, cv2.CAP_FFMPEG)
            if cap.isOpened():
                logging.info(f"Successfully connected to camera on attempt {attempt + 1}")
                logging.info(f"Stream: FPS={cap.get(cv2.CAP_PROP_FPS)}, "
                             f"Width={cap.get(cv2.CAP_PROP_FRAME_WIDTH)}, "
                             f"Height={cap.get(cv2.CAP_PROP_FRAME_HEIGHT)}")
                return cap
            else:
                logging.warning(f"Failed to connect to camera on attempt {attempt + 1}. Retrying in {delay} seconds...")
                cap.release()
                time.sleep(delay)
        except Exception as e:
            logging.error(f"Connection attempt {attempt + 1} failed: {e}")
            time.sleep(delay)
    logging.error("Failed to connect to camera after all retries")
    return None

# Connect to camera
cap = connect_camera(rtsp_url, max_retries, retry_delay)
if not cap:
    logging.warning("Falling back to webcam for testing")
    cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)  # Use webcam as fallback (DSHOW for Windows)
    if not cap.isOpened():
        raise SystemExit("Error: Cannot open camera or webcam")

frame_count = 0
failed_frame_count = 0
target_size = (640, 640)  # Square resolution for MediaPipe

try:
    with mp_face_mesh.FaceMesh(
        static_image_mode=False,
        max_num_faces=20,
        refine_landmarks=True,
        min_detection_confidence=0.4,  # Relaxed for multi-face detection
        min_tracking_confidence=0.4
    ) as face_mesh:
        while True:
            ret, frame = cap.read()
            if not ret or frame is None or frame.size == 0:
                failed_frame_count += 1
                logging.warning(f"Frame capture failed (attempt {failed_frame_count}/{MAX_FAILED_FRAMES})")
                if failed_frame_count >= MAX_FAILED_FRAMES:
                    logging.warning("Max failed frames reached, attempting to reconnect...")
                    cap.release()
                    cap = connect_camera(rtsp_url, max_retries, retry_delay)
                    if not cap:
                        logging.warning("Falling back to webcam for testing")
                        cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)
                        if not cap.isOpened():
                            logging.error("Failed to reconnect to camera or webcam")
                            break
                    failed_frame_count = 0
                continue
            else:
                failed_frame_count = 0  # Reset on successful frame

            # Ensure square frame for MediaPipe
            frame = ensure_square_frame(frame, target_size)
            if frame is None:
                logging.warning("Failed to process square frame, skipping...")
                continue
            logging.debug(f"Frame size: {frame.shape}")

            # Preprocess frame for robust detection
            frame = cv2.convertScaleAbs(frame, alpha=1.3, beta=15)  # Enhance contrast
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            h, w, _ = frame.shape
            lighting_factor = estimate_lighting(frame)
            results = face_mesh.process(rgb)

            if results.multi_face_landmarks:
                for landmarks in results.multi_face_landmarks:
                    # Calculate bounding box and center with robust landmarks
                    valid_landmarks = [p for p in landmarks.landmark[:10] if p.x > 0 and p.y > 0]
                    if len(valid_landmarks) < 5:
                        logging.debug("Insufficient valid landmarks for bounding box")
                        continue
                    xs = [p.x * w for p in valid_landmarks]
                    ys = [p.y * h for p in valid_landmarks]
                    x1, y1, x2, y2 = max(min(xs) - 30, 0), max(min(ys) - 30, 0), min(max(xs) + 30, w), min(max(ys) + 30, h)
                    face_center = ((x1 + x2) / 2, (y1 + y2) / 2)

                    # Determine track ID and initialize liveness detector
                    track_id = None
                    for tid, data in list(tracker.tracked_faces.items()):
                        if np.linalg.norm(np.array(face_center) - np.array(data['center'])) < 200:
                            track_id = tid
                            break
                    if track_id is None:
                        track_id = str(uuid.uuid4())
                        liveness_detectors[track_id] = LivenessDetector()
                        tracker.tracked_faces[track_id] = {
                            'center': face_center,
                            'last_seen': frame_count,
                            'live_frames': 0
                        }

                    # Liveness check
                    is_live, confidence = liveness_detectors[track_id].is_live_face(landmarks, frame.shape, lighting_factor)
                    tracker.update(face_center, is_live, frame_count)

                    # Draw visualization
                    color = (0, 255, 0) if is_live else (0, 0, 255)
                    label = f"ID: {track_id[:8]} {'Live' if is_live else 'Fake'} ({confidence:.2f})"
                    cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), color, 2)
                    cv2.putText(frame, label, (int(x1), int(y1) - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)

                    mp_drawing.draw_landmarks(
                        image=frame,
                        landmark_list=landmarks,
                        connections=mp_face_mesh.FACEMESH_TESSELATION,
                        landmark_drawing_spec=mp_drawing.DrawingSpec(color=(255, 255, 255), thickness=1, circle_radius=1),
                        connection_drawing_spec=mp_drawing.DrawingSpec(color=(0, 255, 255), thickness=1)
                    )
                    mp_drawing.draw_landmarks(
                        image=frame,
                        landmark_list=landmarks,
                        connections=mp_face_mesh.FACEMESH_IRISES,
                        landmark_drawing_spec=mp_drawing.DrawingSpec(color=(255, 0, 0), thickness=1, circle_radius=2),
                        connection_drawing_spec=mp_drawing.DrawingSpec(color=(255, 0, 0), thickness=1)
                    )

            # Draw headcount
            cv2.rectangle(frame, (10, 10), (280, 50), (0, 0, 0), -1)
            cv2.putText(frame, f"Live Headcount: {tracker.get_live_count()}", (20, 40), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)

            cv2.imshow("CCTV Head Count Tracker", frame)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

            frame_count += 1

except Exception as e:
    logging.error(f"Main loop error: {e}")
finally:
    cap.release()
    cv2.destroyAllWindows()