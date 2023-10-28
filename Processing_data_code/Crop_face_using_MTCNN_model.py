import os
import cv2
import glob
from mtcnn import MTCNN

image_dir = "C:\\Users\\phuon\\Downloads\\image_dataset"
image_dir2 = "C:\\Users\\phuon\\Downloads\\image_dataset_cropface"
detector = MTCNN()

image_files = glob.glob(os.path.join(image_dir, "*.jpg"))

for image_path in image_files:
    image = cv2.imread(image_path)
    
    # Sử dụng MTCNN để phát hiện khuôn mặt trong ảnh
    faces = detector.detect_faces(image)

    for i, face in enumerate(faces):
        x, y, width, height = face['box']  # Lấy thông tin về khuôn mặt đã phát hiện

        # Cắt và lưu khuôn mặt đã phát hiện
        cropped_face = image[y:y+height, x:x+width]
        output_filename = os.path.basename(image_path).split('.')[0] + f'_face_{i}.jpg'
        output_path = os.path.join(image_dir2, output_filename)
        cv2.imwrite(output_path, cropped_face)
