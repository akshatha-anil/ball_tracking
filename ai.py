import cv2
import numpy as np

def define_quadrants(frame):
    height, width, _ = frame.shape
    quadrants = {
        1: ((0, width // 2), (0, height // 2)),
        2: ((width // 2, width), (0, height // 2)),
        3: ((0, width // 2), (height // 2, height)),
        4: ((width // 2, width), (height // 2, height))
    }
    return quadrants

def draw_quadrants(frame, quadrants):
    for q, ((x1, x2), (y1, y2)) in quadrants.items():
        cv2.rectangle(frame, (x1, y1), (x2, y2), (255, 255, 255), 2)
        cv2.putText(frame, f"Q{q}", (x1 + 10, y1 + 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
    return frame

def get_quadrant(x, y, quadrants):
    for q, ((x1, x2), (y1, y2)) in quadrants.items():
        if x1 <= x <= x2 and y1 <= y <= y2:
            return q
    return None

def track_balls(video_path):
    cap = cv2.VideoCapture(video_path)
    output_path = 'processed_video.avi'
    fourcc = cv2.VideoWriter_fourcc(*'XVID')
    out = cv2.VideoWriter(output_path, fourcc, 20.0, (int(cap.get(3)), int(cap.get(4))))
    
    log_path = 'event_log.txt'
    with open(log_path, 'w') as log_file:
        log_file.write("Time, Quadrant Number, Ball Colour, Event Type\n")
    
    quadrants = None
    ball_positions = {}
    
    frame_count = 0
    fps = cap.get(cv2.CAP_PROP_FPS)
    
    while(cap.isOpened()):
        ret, frame = cap.read()
        if not ret:
            break
        
        if quadrants is None:
            quadrants = define_quadrants(frame)
        
        hsv_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        
        colors = {
            "red": ((0, 100, 100), (10, 255, 255)),
            "green": ((50, 100, 100), (70, 255, 255)),
            "blue": ((100, 150, 0), (140, 255, 255)),
            "white": ((0, 0, 210), (180, 30, 255))  # Further refined white color range
        }
        
        for color_name, (lower, upper) in colors.items():
            mask = cv2.inRange(hsv_frame, lower, upper)
            
            # Apply morphological operations to filter out noise
            kernel = np.ones((5, 5), np.uint8)
            mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
            mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
            
            contours, _ = cv2.findContours(mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
            
            for cnt in contours:
                area = cv2.contourArea(cnt)
                if area > 500:  # filter out small objects
                    x, y, w, h = cv2.boundingRect(cnt)
                    
                    # Filter out contours that are not roughly circular
                    aspect_ratio = float(w) / h
                    if 0.75 <= aspect_ratio <= 1.25:
                        centroid_x, centroid_y = x + w // 2, y + h // 2
                        quadrant = get_quadrant(centroid_x, centroid_y, quadrants)
                        
                        if color_name not in ball_positions:
                            ball_positions[color_name] = None
                        
                        if ball_positions[color_name] != quadrant:
                            event_type = "Entry" if ball_positions[color_name] is None else "Exit"
                            timestamp = frame_count / fps
                            log_entry = f"{timestamp:.2f}, {quadrant}, {color_name}, {event_type}\n"
                            with open(log_path, 'a') as log_file:
                                log_file.write(log_entry)
                            
                            cv2.putText(frame, f"{event_type} - {timestamp:.2f}s", (x, y + h + 20), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
                            ball_positions[color_name] = quadrant

                        cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
                        cv2.putText(frame, color_name, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)
        
        # Draw quadrants after detecting balls to avoid interference
        frame = draw_quadrants(frame, quadrants)
        out.write(frame)
        frame_count += 1
    
    cap.release()
    out.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    video_path = "input.mp4"
    track_balls(video_path)
    print("Processing complete. Processed video saved as 'processed_video.avi' and log saved as 'event_log.txt'.")
