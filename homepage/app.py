from flask import Flask, Response, render_template, send_file, request, jsonify
import cv2
import mediapipe as mp
import numpy as np
import time
import io
from pose_utils import detectPose, classifyPose
import threading

app = Flask(__name__)

# MediaPipe Pose 초기화
mp_pose = mp.solutions.pose
pose_video = mp_pose.Pose(static_image_mode=False, min_detection_confidence=0.5, model_complexity=1)
mp_drawing = mp.solutions.drawing_utils

# 이미지 로딩 함수
def load_and_resize_image(path, size=(640, 640)):
    image = cv2.imread(path)
    if image is None:
        print(f"Warning: Image not found at {path}. Using blank image instead.")
        return np.zeros(size + (3,), dtype=np.uint8)
    return cv2.resize(image, size)

# 이미지 로딩
pose_images = {
    1: load_and_resize_image('static/stage1.jpg'),
    2: load_and_resize_image('static/stage2.jpg'),
    3: load_and_resize_image('static/stage3.jpg'),
    4: load_and_resize_image('static/stage4.jpg'),
    5: load_and_resize_image('static/stage5.jpg'),
    6: load_and_resize_image('static/stage6.jpg'),
    7: load_and_resize_image('static/stage7.jpg'),
    8: load_and_resize_image('static/finish.png')
}

pose_durations = {
    1: {"pose": "Right Stretch waist", "image": pose_images[1], "next_stage": 2},
    2: {"pose": "Left Stretch waist", "image": pose_images[2], "next_stage": 3},
    3: {"pose": "Right Lunge Pose", "image": pose_images[3], "next_stage": 4},
    4: {"pose": "Left Lunge Pose", "image": pose_images[4], "next_stage": 5},
    5: {"pose": "Squat Pose", "image": pose_images[5], "next_stage": 6},
    6: {"pose": "Tree Pose", "image": pose_images[6], "next_stage": 7},
    7: {"pose": "Tree Pose", "image": pose_images[7], "next_stage": 8},
    8: {"pose": "Finished", "image": pose_images[8], "next_stage": None}
}

pose_held_duration = 3  # 각 포즈를 유지해야 하는 시간(초)
current_stage = 1  # 초기 포즈 스테이지
pose_start_time = None  # 포즈 시작 시간을 전역 변수로 설정
retry_count = 0
max_retries = 5
detect_pose = True

# 카메라 초기화
camera = cv2.VideoCapture(0)
frame_lock = threading.Lock()

def gen_frames():
    global current_stage, pose_start_time, retry_count, detect_pose
    while True:
        success, frame = camera.read()
        if not success:
            retry_count += 1
            if retry_count >= max_retries:
                print("Failed to read from camera after multiple attempts.")
                break
            continue
        retry_count = 0

        with frame_lock:
            frame = cv2.flip(frame, 1)

            if detect_pose:
                frame, landmarks = detectPose(frame, pose_video, display=False)
                if landmarks:
                    frame, pose_label = classifyPose(landmarks, frame, display=False)

                    current_pose = pose_durations[current_stage]["pose"]

                    if pose_label == current_pose:
                        if pose_start_time is None:
                            pose_start_time = time.time()
                        elapsed_time = time.time() - pose_start_time
                        remaining_time = max(0, pose_held_duration - elapsed_time)

                        if elapsed_time >= pose_held_duration:
                            pose_start_time = None
                            next_stage = pose_durations[current_stage]["next_stage"]
                            if next_stage is None:
                                detect_pose = False
                            else:
                                current_stage = next_stage

        _, buffer = cv2.imencode('.jpg', frame)
        frame = buffer.tobytes()

        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')

@app.route('/timer_status')
def timer_status():
    global pose_start_time
    global current_stage
    global detect_pose

    if pose_start_time is not None:
        elapsed_time = time.time() - pose_start_time
        remaining_time = max(0, pose_held_duration - elapsed_time)
    else:
        remaining_time = pose_held_duration

    if detect_pose:
        success, frame = camera.read()
        if success:
            frame = cv2.flip(frame, 1)
            frame, landmarks = detectPose(frame, pose_video, display=False)
            if landmarks:
                _, pose_label = classifyPose(landmarks, frame, display=False)
            else:
                pose_label = "Unknown Pose"
        else:
            pose_label = "Camera Error"
    else:
        pose_label = "Done"

    return jsonify({'stage': current_stage, 'remaining_time': remaining_time, 'pose_label': pose_label})

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/video_feed')
def video_feed():
    return Response(gen_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/pose_image')
def pose_image():
    stage = int(request.args.get('stage', 1))
    if stage in pose_durations:
        pose_image = pose_durations[stage]["image"]
        _, buffer = cv2.imencode('.jpg', pose_image)
        return send_file(io.BytesIO(buffer), mimetype='image/jpeg')
    else:
        return "Image not found", 404

@app.route('/current_stage')
def current_stage_route():
    return jsonify({'stage': current_stage})

@app.route('/update_stage', methods=['POST'])
def update_stage():
    global current_stage
    global pose_start_time
    new_stage = request.json.get('stage')
    if new_stage is not None and new_stage in pose_durations:
        current_stage = new_stage
        pose_start_time = None
        return jsonify({'stage': current_stage})
    return jsonify({'error': 'Invalid stage'}), 400

# 추가된 라우트

@app.route('/next_page')
def next_page():
    return render_template('next_page.html')

@app.route('/pose_page')
def pose_page():
    return render_template('pose_page.html')

@app.route('/reset_stage', methods=['POST'])
def reset_stage():
    global current_stage
    global pose_start_time
    current_stage = 1  # 초기 스테이지로 리셋
    pose_start_time = None
    return jsonify({'stage': current_stage})

if __name__ == '__main__':
    try:
        app.run(debug= False)
    finally:
        camera.release()  # 앱 종료 시 카메라 해제