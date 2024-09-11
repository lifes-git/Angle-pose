import math
import cv2
import numpy as np
import mediapipe as mp
from PIL import Image, ImageSequence

mp_pose = mp.solutions.pose
pose = mp_pose.Pose(static_image_mode=True, min_detection_confidence=0.3, model_complexity=2)
mp_drawing = mp.solutions.drawing_utils

def detectPose(image, pose, display=True):
    output_image = image.copy()
    imageRGB = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    results = pose.process(imageRGB)
    height, width, _ = image.shape
    landmarks = []
    
    if results.pose_landmarks:
        #mp_drawing.draw_landmarks(image=output_image, landmark_list=results.pose_landmarks,
                                  #connections=mp_pose.POSE_CONNECTIONS)
        landmarks = [(int(landmark.x * width), int(landmark.y * height), int(landmark.z * width))
                     for landmark in results.pose_landmarks.landmark]
    
    return output_image, landmarks

def calculateAngle(landmark1, landmark2, landmark3):
    x1, y1, _ = landmark1
    x2, y2, _ = landmark2
    x3, y3, _ = landmark3

    angle = math.degrees(math.atan2(y3 - y2, x3 - x2) - math.atan2(y1 - y2, x1 - x2))
    if angle < 0:
        angle += 360

    return angle

def classifyPose(landmarks, output_image, display=False):
    label = 'Unknown Pose'
    color = (0, 0, 255)  # 알 수 없는 자세를 위한 빨간색

    if len(landmarks) < 33:  # 충분한 랜드마크가 있는지 확인
        return output_image, label

    try:
        left_elbow_angle = calculateAngle(landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value],
                                          landmarks[mp_pose.PoseLandmark.LEFT_ELBOW.value],
                                          landmarks[mp_pose.PoseLandmark.LEFT_WRIST.value])
        
        right_elbow_angle = calculateAngle(landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER.value],
                                           landmarks[mp_pose.PoseLandmark.RIGHT_ELBOW.value],
                                           landmarks[mp_pose.PoseLandmark.RIGHT_WRIST.value])
        
        left_shoulder_angle = calculateAngle(landmarks[mp_pose.PoseLandmark.LEFT_ELBOW.value],
                                             landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value],
                                             landmarks[mp_pose.PoseLandmark.LEFT_HIP.value])
        
        right_shoulder_angle = calculateAngle(landmarks[mp_pose.PoseLandmark.RIGHT_HIP.value],
                                              landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value],
                                              landmarks[mp_pose.PoseLandmark.RIGHT_ELBOW.value])
        
        left_knee_angle = calculateAngle(landmarks[mp_pose.PoseLandmark.LEFT_HIP.value],
                                         landmarks[mp_pose.PoseLandmark.LEFT_KNEE.value],
                                         landmarks[mp_pose.PoseLandmark.LEFT_ANKLE.value])
        
        right_knee_angle = calculateAngle(landmarks[mp_pose.PoseLandmark.RIGHT_HIP.value],
                                          landmarks[mp_pose.PoseLandmark.RIGHT_KNEE.value],
                                          landmarks[mp_pose.PoseLandmark.RIGHT_ANKLE.value])
       
        # 자세를 각도에 따라 분류
        if (165 < left_knee_angle < 195) or (165 < right_knee_angle < 195):  # 트리 포즈
            if (310 < left_knee_angle < 335) or (30 < right_knee_angle < 60):
                label = 'Tree Pose'

        elif (180 < left_knee_angle < 220) and (150 < right_knee_angle < 190):  # 스쿼트 포즈
            label = 'Squat Pose'

        elif (250 < left_knee_angle < 290) and (250 < right_knee_angle < 290):  # 오른쪽 런지 포즈
            label = 'Right Lunge Pose'

        elif (80 < left_knee_angle < 120) and (80 < right_knee_angle < 120):  # 왼쪽 런지 포즈
            label = 'Left Lunge Pose'

        if (150 < left_elbow_angle < 190) and (180 < right_elbow_angle < 230):
            if (170 < left_shoulder_angle < 230) and (70 < right_shoulder_angle < 100):
                label = 'Left Stretch waist'  # 팔이 펼쳐진 포즈 확인 1
            
            if (140 < left_shoulder_angle < 190) and (110 < right_shoulder_angle < 150):
                label = 'Right Stretch waist'  # 팔이 펼쳐진 포즈 확인 2
            
    except IndexError:
        # 랜드마크가 부족하여 인덱스 오류 발생 시 처리
        pass

    if label != 'Unknown Pose':
        color = (0, 255, 0)  # 자세가 식별된 경우 녹색

    return output_image, label

def flip_frames(frames):
    flipped_frames = [cv2.flip(frame, 1) for frame in frames]  # 프레임을 수평으로 뒤집기
    return flipped_frames
