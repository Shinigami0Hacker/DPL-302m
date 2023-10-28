import cv2
import os
import glob

source_folder_path = "C:\\Users\\phuon\\Downloads\\Emo_Clips"
save_folder_path = "C:\\Users\\phuon\\Downloads\\dataset"

if not os.path.exists(save_folder_path):
    os.makedirs(save_folder_path)

frame_count_anger=0
frame_count_disgust=0
frame_count_happy=0
frame_count_surprise=0
frame_count_sad=0
frame_count_fear=0
frame_count_neutral=0

video_files = glob.glob(source_folder_path + "/*.mp4")

for video_file in video_files:
    cap = cv2.VideoCapture(video_file)
    if not cap.isOpened():
        print("Không thể mở video: " + video_file)
        continue
    
    video_name = os.path.basename(video_file)
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        if 'A' in video_name:
            frame_count_anger+=1
            image_path = os.path.join(save_folder_path, "anger" + str(frame_count_anger) + ".jpg")
            cv2.imwrite(image_path, frame)
        elif 'D' in video_name:
            frame_count_disgust+=1
            image_path = os.path.join(save_folder_path, "disgust" + str(frame_count_disgust) + ".jpg")
            cv2.imwrite(image_path, frame)
        elif 'F' in video_name:
            frame_count_fear+=1
            image_path = os.path.join(save_folder_path, "fear" + str(frame_count_fear) + ".jpg")
            cv2.imwrite(image_path, frame)
        elif 'H' in video_name:
            frame_count_happy+=1
            image_path = os.path.join(save_folder_path, "happy" + str(frame_count_happy) + ".jpg")
            cv2.imwrite(image_path, frame)
        elif 'N' in video_name:
            frame_count_neutral+=1
            image_path = os.path.join(save_folder_path, "neutral" + str(frame_count_neutral) + ".jpg")
            cv2.imwrite(image_path, frame)
        elif 'S' and 'U' in video_name:
            frame_count_surprise+=1
            image_path = os.path.join(save_folder_path, "surprise" + str(frame_count_surprise) + ".jpg")
            cv2.imwrite(image_path, frame)
        else:
            frame_count_sad+=1
            image_path = os.path.join(save_folder_path, "sad" + str(frame_count_sad) + ".jpg")
            cv2.imwrite(image_path, frame)
    cap.release()

