import cv2
import mediapipe as mp
import numpy as np
import math
from collections import defaultdict
from ultralytics import YOLO

class Person:
    """Class to track a single person's attention metrics"""
    def __init__(self, person_id):
        self.id = person_id
        self.concentration_history = []
        self.max_history = 30
        self.phone_detected = False
        self.phone_timer = 0
        self.last_bbox = None
        self.frames_not_detected = 0
    
    def update_concentration(self, head_down, head_turned, phone):
        """Update concentration score for this person"""
        # Head down is GOOD (high concentration) - weight 0.7
        # Head turned is BAD (low concentration) - weight 0.3
        # Phone is VERY BAD - extreme penalty
        
        # Calculate base posture score
        posture_score = (head_down * 0.7) - (head_turned * 0.3)
        posture_score = max(0.0, min(1.0, posture_score))
        
        # Apply phone penalty (VERY HARSH - immediate drop to near 0)
        if phone:
            posture_score = posture_score * 0.1  # 90% penalty if phone detected - phone = distraction
        
        # Convert to 0-100 scale
        concentration = posture_score * 100
        concentration = max(0, min(100, concentration))
        
        self.concentration_history.append(concentration)
        if len(self.concentration_history) > self.max_history:
            self.concentration_history.pop(0)
        
        return concentration
    
    def get_avg_concentration(self):
        """Get average concentration for this person"""
        if not self.concentration_history:
            return 50
        return np.mean(self.concentration_history)


class AttentionMonitor:
    def __init__(self):
        # Initialize MediaPipe components
        self.mp_pose = mp.solutions.pose
        self.mp_face_detection = mp.solutions.face_detection
        self.mp_drawing = mp.solutions.drawing_utils
        
        self.pose = self.mp_pose.Pose(
            static_image_mode=False,
            model_complexity=1,
            smooth_landmarks=True,
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5
        )
        
        self.face_detector = self.mp_face_detection.FaceDetection(
            model_selection=0,
            min_detection_confidence=0.5
        )
        
        # Initialize YOLOv8 for phone detection
        print("Loading YOLOv8 model for phone detection...")
        self.yolo_model = YOLO('yolov8n.pt')  # Load nano model (fast)
        self.phone_class_id = 67  # Class ID for cell phone in COCO dataset
        print("✅ YOLOv8 model loaded successfully")
        
        # Multi-person tracking
        self.people = {}  # Dictionary to store Person objects by ID
        self.next_person_id = 0
        self.global_phone_detected = False
        self.detected_phones = []  # List of detected phone bounding boxes
        self.frame_count = 0  # Frame counter for debugging
        
        # Phone detection state
        self.detected_phones = []
    
    def detect_phones(self, frame):
        """Detect phones in the frame using YOLOv8"""
        try:
            # Run YOLO inference
            results = self.yolo_model(frame, verbose=False, conf=0.5)
            
            self.detected_phones = []
            
            # Extract phone detections (class 67 = cell phone in COCO)
            for result in results:
                boxes = result.boxes
                for box in boxes:
                    class_id = int(box.cls[0])
                    conf = float(box.conf[0])
                    
                    # Detection: cell phone (67), mobile phone, iPhone, etc.
                    if class_id == 67 and conf > 0.3:  # Cell phone - lower threshold for better detection
                        x1, y1, x2, y2 = map(int, box.xyxy[0])
                        self.detected_phones.append({
                            'bbox': (x1, y1, x2, y2),
                            'conf': conf,
                            'center': ((x1 + x2) // 2, (y1 + y2) // 2)
                        })
                        if self.frame_count % 30 == 0:
                            print(f"📱 Phone detected at confidence {conf:.2f}")
            
            return len(self.detected_phones) > 0
        
        except Exception as e:
            print(f"Phone detection error: {e}")
            return False
    
    def check_phone_near_person(self, person_face_bbox):
        """Check if a phone is detected near a person's face"""
        if not self.detected_phones:
            return False
        
        face_x_min, face_y_min, face_x_max, face_y_max = person_face_bbox
        face_center = ((face_x_min + face_x_max) // 2, (face_y_min + face_y_max) // 2)
        
        # Check if any detected phone is near this person's face
        for phone in self.detected_phones:
            phone_center = phone['center']
            distance = np.sqrt((phone_center[0] - face_center[0])**2 + 
                             (phone_center[1] - face_center[1])**2)
            
            # If phone is within 300 pixels of person's face
            if distance < 300:
                if self.frame_count % 30 == 0:
                    print(f"⚠️ Phone is close to person's face! Distance: {distance:.0f}px, Confidence: {phone['conf']:.2f}")
                return True
        
        return False
    
    def calculate_angle(self, point1, point2, point3):
        """Calculate angle between three points"""
        x1, y1 = point1
        x2, y2 = point2
        x3, y3 = point3
        
        angle = math.atan2(y3 - y2, x3 - x2) - math.atan2(y1 - y2, x1 - x2)
        angle = math.degrees(angle)
        return angle % 360
    
    def calculate_head_posture(self, landmarks):
        """
        Calculate head posture metrics
        Returns: head_down_score (0-1), head_turned_score (0-1)
        """
        try:
            # Handle NormalizedLandmarkList - convert to list of landmarks
            if hasattr(landmarks, 'landmark'):
                # It's a NormalizedLandmarkList, get the actual list
                landmark_list = list(landmarks.landmark)
            else:
                # It's already a list
                landmark_list = landmarks if isinstance(landmarks, list) else []
            
            if not landmark_list or len(landmark_list) < 13:  # Need key points
                return 0.5, 0.5
            
            # Key landmarks from MediaPipe Pose
            nose = np.array([landmark_list[0].x, landmark_list[0].y])
            left_eye = np.array([landmark_list[2].x, landmark_list[2].y])
            right_eye = np.array([landmark_list[5].x, landmark_list[5].y])
            
            # Get visibility scores
            nose_vis = landmark_list[0].visibility if hasattr(landmark_list[0], 'visibility') else 1.0
            left_eye_vis = landmark_list[2].visibility if hasattr(landmark_list[2], 'visibility') else 1.0
            right_eye_vis = landmark_list[5].visibility if hasattr(landmark_list[5], 'visibility') else 1.0
            
            # If key landmarks not visible, return neutral
            if nose_vis < 0.3 or left_eye_vis < 0.3 or right_eye_vis < 0.3:
                return 0.5, 0.5
            
            # Calculate eyes center
            eye_center = (left_eye + right_eye) / 2
            eye_distance = np.linalg.norm(right_eye - left_eye)
            
            # Safety check
            if eye_distance < 0.01:
                return 0.5, 0.5
            
            # 1. HEAD DOWN detection - nose below eyes = head down
            # Positive y_diff means nose is below eyes (head looking down)
            y_diff = nose[1] - eye_center[1]
            # Normalize by eye distance
            head_down_score = min(1.0, max(0.0, (y_diff + 0.05) / 0.15))
            
            # 2. HEAD TURNED detection - nose offset from eye center = head turned
            # If nose x is far from eye center x, head is turned
            x_diff = abs(nose[0] - eye_center[0])
            # Normalize by eye distance
            head_turned_score = min(1.0, x_diff / (eye_distance * 0.5))
            
            return float(head_down_score), float(head_turned_score)
        
        except Exception as e:
            print(f"❌ Error in posture calc: {e}")
            return 0.5, 0.5
    
    def calculate_concentration_score(self, head_down_score, head_turned_score, phone_detected):
        """
        Calculate overall concentration score (0-100)
        High score = high concentration
        """
        # Head down is good (high concentration)
        # Head turned is bad (low concentration)
        posture_score = (head_down_score * 0.6 - head_turned_score * 0.4)
        posture_score = max(0, min(1, posture_score))
        
        # Phone detection reduces concentration
        phone_penalty = 0.5 if phone_detected else 0.0
        
        # Final score
        concentration = (posture_score * (1 - phone_penalty)) * 100
        concentration = max(0, min(100, concentration))
        
        return concentration
    
    def get_posture_label(self, head_down, head_turned):
        """Get human-readable posture label"""
        if head_turned > 0.3:
            return "Looking Away"
        elif head_down > 0.4:
            return "Focused"
        else:
            return "Neutral"
    
    def process_frame(self, frame):
        """Process a frame and return annotated frame with metrics"""
        self.frame_count += 1
        h, w, c = frame.shape
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        
        # Detect phones using YOLO
        phone_detected = self.detect_phones(frame)
        
        # Detect all faces
        face_results = self.face_detector.process(rgb_frame)
        
        # Get global pose for all people in frame
        pose_results = self.pose.process(rgb_frame)
        
        # Reset frames_not_detected counter
        for person in self.people.values():
            person.frames_not_detected += 1
        
        # Dictionary to map detected faces to landmarks
        detected_faces = []
        if face_results.detections:
            for detection in face_results.detections:
                bboxC = detection.location_data.relative_bounding_box
                x_min = int(bboxC.xmin * w)
                y_min = int(bboxC.ymin * h)
                x_max = int((bboxC.xmin + bboxC.width) * w)
                y_max = int((bboxC.ymin + bboxC.height) * h)
                
                # Add padding to face box
                padding = int((x_max - x_min) * 0.1)
                x_min = max(0, x_min - padding)
                y_min = max(0, y_min - padding)
                x_max = min(w, x_max + padding)
                y_max = min(h, y_max + padding)
                
                detected_faces.append({
                    'bbox': (x_min, y_min, x_max, y_max),
                    'center': ((x_min + x_max) // 2, (y_min + y_max) // 2),
                    'area': (x_max - x_min) * (y_max - y_min)
                })
        
        # Track existing people and match with faces
        matched_faces = set()
        for person_id, person in list(self.people.items()):
            if person.last_bbox and detected_faces:
                # Find closest face to this person's last location
                best_match = None
                best_distance = float('inf')
                
                for face_idx, face in enumerate(detected_faces):
                    if face_idx in matched_faces:
                        continue
                    
                    # Calculate distance between centers
                    last_center = ((person.last_bbox[0] + person.last_bbox[2]) // 2,
                                   (person.last_bbox[1] + person.last_bbox[3]) // 2)
                    distance = np.sqrt((face['center'][0] - last_center[0])**2 + 
                                     (face['center'][1] - last_center[1])**2)
                    
                    if distance < best_distance and distance < 100:  # 100 pixel threshold
                        best_distance = distance
                        best_match = face_idx
                
                if best_match is not None:
                    matched_faces.add(best_match)
                    person.frames_not_detected = 0
                else:
                    # Person not found, increment counter
                    if person.frames_not_detected > 10:
                        del self.people[person_id]
                        continue
            
            # Update phone timer
            if person.phone_detected:
                person.phone_timer -= 1
                if person.phone_timer <= 0:
                    person.phone_detected = False
            
            # Update global phone state
            if person.phone_detected:
                self.global_phone_detected = True
        
        self.global_phone_detected = False  # Reset global phone state
        
        # Add new people for unmatched faces
        for face_idx, face in enumerate(detected_faces):
            if face_idx not in matched_faces:
                # New person detected
                person_id = self.next_person_id
                self.next_person_id += 1
                self.people[person_id] = Person(person_id)
                self.people[person_id].last_bbox = face['bbox']
                matched_faces.add(face_idx)
        
        # Process pose landmarks for all detected poses
        pose_landmarks_list = []
        has_pose = False
        if pose_results and hasattr(pose_results, 'pose_landmarks') and pose_results.pose_landmarks:
            pose_landmarks_list = [pose_results.pose_landmarks]
            has_pose = True
            if self.frame_count % 30 == 0:
                # Handle NormalizedLandmarkList - get count of actual landmarks
                landmark_count = len(list(pose_results.pose_landmarks.landmark)) if hasattr(pose_results.pose_landmarks, 'landmark') else len(pose_results.pose_landmarks)
                print(f"✅ Pose detected! Landmarks count: {landmark_count}")
        else:
            if self.frame_count % 30 == 0:
                print(f"⚠️ NO pose detected on frame {self.frame_count} - using default scores")
            # Don't return early - continue processing with default scores
        
        # Draw faces and calculate concentration for each detected person
        person_idx = 0
        for face_idx, face in enumerate(detected_faces):
            x_min, y_min, x_max, y_max = face['bbox']
            
            # Draw face bounding box
            cv2.rectangle(frame, (x_min, y_min), (x_max, y_max), (0, 255, 0), 2)
            
            # Find corresponding person ID
            person_id = None
            for pid, person in self.people.items():
                if person.last_bbox == face['bbox']:
                    person_id = pid
                    break
            
            if person_id is None:
                continue
            
            person = self.people[person_id]
            person.last_bbox = face['bbox']
            
            # Get pose landmarks - use the first available pose or default scores
            head_down_score = 0.5
            head_turned_score = 0.5
            
            if has_pose and pose_landmarks_list and person_idx < len(pose_landmarks_list):
                landmarks_raw = pose_landmarks_list[person_idx]
                
                try:
                    # Calculate head posture (automatically handles NormalizedLandmarkList)
                    head_down_score, head_turned_score = self.calculate_head_posture(landmarks_raw)
                    
                    # Convert to list for drawing - absolutely ensure we have a list
                    if hasattr(landmarks_raw, 'landmark'):
                        landmarks = list(landmarks_raw.landmark)
                    else:
                        # Try to convert to list even if it's not a NormalizedLandmarkList
                        try:
                            landmarks = list(landmarks_raw)
                        except:
                            landmarks = []
                    
                    # Draw pose skeleton (upper body)
                    key_points = [0, 2, 5, 11, 12, 13, 14]
                    for idx in key_points:
                        if idx < len(landmarks) and landmarks[idx].visibility > 0.5:
                            x = int(landmarks[idx].x * w)
                            y = int(landmarks[idx].y * h)
                            cv2.circle(frame, (x, y), 5, (255, 0, 0), -1)
                    
                    # Draw connections
                    connections = [(11, 13), (13, 15), (12, 14), (14, 16), (11, 12)]
                    for start, end in connections:
                        if start < len(landmarks) and end < len(landmarks):
                            if landmarks[start].visibility > 0.5 and landmarks[end].visibility > 0.5:
                                x1 = int(landmarks[start].x * w)
                                y1 = int(landmarks[start].y * h)
                                x2 = int(landmarks[end].x * w)
                                y2 = int(landmarks[end].y * h)
                                cv2.line(frame, (x1, y1), (x2, y2), (255, 0, 0), 2)
                
                except Exception as e:
                    pass
                
                person_idx += 1
            else:
                # No pose landmarks - use default neutral scores
                head_down_score = 0.5
                head_turned_score = 0.5
            
            try:
                # Check if phone is near this person's face
                person.phone_detected = self.check_phone_near_person(face['bbox'])
                
                concentration = person.update_concentration(head_down_score, head_turned_score, 
                                                          person.phone_detected)
                
                # Debug: Print once per person per second
                if self.frame_count % 30 == 0:
                    avg_conc = person.get_avg_concentration()
                    phone_status = "📱 PHONE DETECTED" if person.phone_detected else "✓ OK"
                    print(f"P{person.id}: Down={head_down_score:.2f}, Turn={head_turned_score:.2f}, Conc={avg_conc:.1f}% ({phone_status})")
                
                # Draw person info on frame
                self._draw_person_info(frame, person, face, head_down_score, head_turned_score)
            
            except Exception as e:
                print(f"Error processing person: {e}")
        
        # Draw detected phones
        self._draw_detected_phones(frame)
        
        # Draw global stats
        self._draw_global_stats(frame)
        
        return frame
    
    def _draw_person_info(self, frame, person, face, head_down, head_turned):
        """Draw information for a single person"""
        x_min, y_min, x_max, y_max = face['bbox']
        h, w = frame.shape[:2]
        
        # Determine concentration color
        avg_conc = person.get_avg_concentration()
        if avg_conc > 70:
            conc_color = (0, 255, 0)  # Green
        elif avg_conc > 40:
            conc_color = (0, 255, 255)  # Yellow
        else:
            conc_color = (0, 0, 255)  # Red
        
        # Draw person ID on face box
        cv2.putText(frame, f"ID: {person.id}", (x_min, y_min - 10), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        
        # Draw concentration score above the face
        cv2.putText(frame, f"Conc: {avg_conc:.0f}", (x_min, y_min - 35), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, conc_color, 2)
        
        # Draw debug info: head down and turned scores
        debug_text = f"Down:{head_down:.2f} Turn:{head_turned:.2f}"
        cv2.putText(frame, debug_text, (x_min, y_min - 60), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)
        
        # Draw small bar under the face
        bar_width = x_max - x_min
        bar_height = 5
        bar_y = y_max + 5
        
        # Background bar
        cv2.rectangle(frame, (x_min, bar_y), (x_max, bar_y + bar_height), 
                     (0, 0, 255), -1)
        
        # Foreground bar
        filled_width = int(bar_width * (avg_conc / 100))
        if filled_width > 0:
            cv2.rectangle(frame, (x_min, bar_y), (x_min + filled_width, bar_y + bar_height), 
                         conc_color, -1)
    
    def _draw_detected_phones(self, frame):
        """Draw detected phones on frame with red bounding boxes"""
        if not self.detected_phones:
            return
        
        for phone in self.detected_phones:
            x1, y1, x2, y2 = phone['bbox']
            conf = phone['conf']
            
            # Draw red bounding box for phone
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 0, 255), 3)
            
            # Draw label with confidence
            label = f"Phone {conf*100:.0f}%"
            cv2.putText(frame, label, (x1, y1 - 10), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
            
            # Draw a warning indicator
            cv2.putText(frame, "📱 PHONE DETECTED", (x1, y2 + 25), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
    
    def _draw_global_stats(self, frame):
        """Draw global statistics on frame"""
        h, w = frame.shape[:2]
        y_offset = 30
        
        # Title
        cv2.putText(frame, f"Attention Monitor - {len(self.people)} Person(s)", (10, y_offset), 
                   cv2.FONT_HERSHEY_SIMPLEX, 1.2, (255, 255, 255), 2)
        y_offset += 40
        
        if self.people:
            # Calculate average concentration across all people
            all_concentrations = [p.get_avg_concentration() for p in self.people.values()]
            avg_all = np.mean(all_concentrations)
            
            cv2.putText(frame, f"Global Avg: {avg_all:.0f}", (10, y_offset), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.8, (200, 200, 200), 2)
            y_offset += 30
            
            # List each person's stats
            for person in self.people.values():
                avg_conc = person.get_avg_concentration()
                status = "Focused" if avg_conc > 60 else "Distracted" if avg_conc < 40 else "Neutral"
                conc_color = (0, 255, 0) if avg_conc > 70 else (0, 255, 255) if avg_conc > 40 else (0, 0, 255)
                
                text = f"P{person.id}: {avg_conc:.0f} ({status})"
                cv2.putText(frame, text, (10, y_offset), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.6, conc_color, 1)
                y_offset += 25
        
        # Instructions
        cv2.putText(frame, "Phone detected automatically | Q=Quit", (10, h - 30), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (200, 200, 200), 1)
    
    def run(self):
        """Main loop to capture and process video"""
        cap = cv2.VideoCapture(0)
        
        if not cap.isOpened():
            print("Error: Cannot access webcam")
            print("Make sure your webcam is connected and not in use by another application")
            return
        
        # Set camera resolution
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
        
        # Check frame capture
        ret, test_frame = cap.read()
        if not ret:
            print("Error: Cannot read frames from webcam")
            cap.release()
            return
        
        print("\n" + "="*60)
        print("MULTI-PERSON ATTENTION MONITOR WITH AUTO PHONE DETECTION")
        print("="*60)
        print("Features:")
        print("  • Automatic phone detection using YOLOv8")
        print("  • Multi-person tracking and concentration scoring")
        print("  • Real-time visual feedback")
        print("\nControls:")
        print("  Q - Quit the application")
        print("\nThe system will track multiple people and estimate their concentration.")
        print("Phones will be detected automatically using AI-powered vision.")
        print("Green bar = High concentration, Red bar = Low concentration")
        print("="*60 + "\n")
        
        frame_count = 0
        try:
            while True:
                ret, frame = cap.read()
                
                if not ret:
                    print("Error: Failed to read frame")
                    break
                
                frame_count += 1
                
                # Flip frame horizontally for selfie view
                frame = cv2.flip(frame, 1)
                
                # Process frame
                try:
                    annotated_frame = self.process_frame(frame)
                    
                    # Display
                    cv2.imshow("Multi-Person Attention Monitor", annotated_frame)
                except Exception as frame_error:
                    print(f"Error processing frame {frame_count}: {frame_error}")
                    break
                
                # Handle keyboard input (1ms wait)
                key = cv2.waitKey(1) & 0xFF
                
                if key == ord('q'):
                    print("\nQuitting...")
                    break
        
        except KeyboardInterrupt:
            print("\nInterrupted by user")
        finally:
            cap.release()
            cv2.destroyAllWindows()
            self.pose.close()
            self.face_detector.close()
            print(f"\nAttention Monitor closed.")
            print(f"Processed {frame_count} frames")
            if self.people:
                print(f"Final stats - Tracked {len(self.people)} person/people")


def main():
    monitor = AttentionMonitor()
    monitor.run()


if __name__ == "__main__":
    main()
