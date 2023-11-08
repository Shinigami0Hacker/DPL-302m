import cv2
import numpy as np
import pandas as pd
from collections import Counter, deque
from tensorflow.keras.models import load_model
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision

class EmotionClassifier:
    def __init__(self, model_path, face_landmarker_path):
        base_options = python.BaseOptions(model_asset_path=face_landmarker_path)

        options = vision.FaceLandmarkerOptions(base_options=base_options,
                                    output_face_blendshapes=True,
                                    output_facial_transformation_matrixes=True,
                                    num_faces=1)
        self.face_landmark_detector = vision.FaceLandmarker.create_from_options(options)
        self.face_detector = mp.solutions.face_detection.FaceDetection(min_detection_confidence=0.5)
        self.model = load_model(model_path)
        self.dataframe=pd.DataFrame()
        self.frame_queue = deque(maxlen=5)
        self.face_coordinates_queue = deque(maxlen=5)
        self.label = None
        self.predictions=None

    def add_bendshapes(self, frame):
        '''
        This function takes an input frame, converts it to a NumPy array, and then uses the MediaPipe library to detect face blendshapes in the frame.

        Parameters:
        - frame: Input frame (e.g., an image or a frame from a video stream).

        Returns:
        - df_temp: A Pandas DataFrame containing the detected face blendshape categories and scores. If no blendshapes are detected, it returns None.
        '''
        numpy_image = np.array(frame)
        image = mp.Image(image_format=mp.ImageFormat.SRGB, data=numpy_image)
        detection_result = self.face_landmark_detector.detect(image)
        category_dict = {}
        if detection_result.face_blendshapes:
            for category_list in detection_result.face_blendshapes:
                for category in category_list:
                    category_dict[category.category_name] = category.score
                    df_temp = pd.DataFrame(category_dict, index=[0])
            return df_temp       
        return None
    
    def add_coordinates(self, frame):
        '''
        This function, add_coordinates, takes an input frame, processes it to detect a face, and returns the coordinates of the detected face's bounding box.

        Parameters:
        - frame: Input frame, typically an image or a frame from a video stream.

        Returns:
        - If a face is detected, it returns a tuple (x, y, w, h) representing the top-left corner coordinates (x, y) and the width (w) and height (h) of the bounding box. If no face is detected, it returns None.
        '''
        image_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = self.face_detector.process(image_rgb)
        if results.detections:
            face_detection_data = results.detections[0]
            bboxC = face_detection_data.location_data.relative_bounding_box
            ih, iw, _ = frame.shape
            x, y, w, h = int(bboxC.xmin * iw), int(bboxC.ymin * ih), int(bboxC.width * iw), int(bboxC.height * ih)
            return (x,y,w,h)
        return None

    def predict_frames(self):
        columns=['mouthSmileRight',
 'mouthSmileLeft',
 'eyeSquintLeft',
 'eyeSquintRight',
 'eyeLookDownRight',
 'mouthUpperUpRight',
 'eyeBlinkLeft',
 'eyeLookDownLeft',
 'mouthUpperUpLeft',
 'eyeBlinkRight',
 'mouthLowerDownRight',
 'eyeLookInRight',
 'eyeLookOutLeft',
 'mouthLowerDownLeft',
 'browOuterUpLeft',
 'eyeLookUpLeft',
 'mouthStretchRight',
 'browOuterUpRight',
 'eyeLookUpRight',
 'eyeLookInLeft',
 'eyeLookOutRight',
 'mouthStretchLeft',
 'browInnerUp',
 'mouthPressRight',
 'browDownRight',
 'mouthPressLeft',
 'browDownLeft',
 'jawOpen',
 'mouthShrugUpper',
 'mouthFunnel',
 'mouthPucker',
 'mouthRollLower',
 'mouthShrugLower',
 'eyeWideLeft',
 'mouthRollUpper',
 'eyeWideRight',
 'mouthFrownRight']
        if self.predictions is None:
            self.predictions = [] 
            for i in range(self.dataframe.shape[0]):
                sample = np.array([self.dataframe[columns].iloc[i]]) 
                prediction = self.model.predict(sample)
                self.predictions.append(prediction)
        else:
            sample = np.array([self.dataframe[columns].iloc[self.dataframe.shape[0]-1]]) 
            prediction = self.model.predict(sample)
            self.predictions.append(prediction)

        prediction_class = self.mapping_class(self.predictions)
        count = Counter(prediction_class)
        self.label, _ = count.most_common(1)[0]
        return self.predictions,self.label
    
    def mapping_class(self,predictions):
        '''
        This function, mapping_class, maps numerical prediction values to their corresponding class labels based on a predefined class mapping.

        Parameters:
        - predictions: A list of numerical predictions.

        Returns:
        - prediction_class: A list of class labels corresponding to the input predictions.
        '''
        class_mapping = {0: "negative",
                        1: "positive",
                        2: "neutral"
                        }
        prediction_class=[]
        for i, prediction in enumerate(predictions):
            predicted_class_index = np.argmax(prediction)
            class_label = class_mapping.get(predicted_class_index, "Unknown")
            prediction_class.append(class_label)
        return prediction_class

    def process_video(self):
        '''
        This function, process_video, captures video from a camera feed, performs real-time processing, including face blendshape detection and class prediction, and displays the processed video.

        It maintains a rolling queue of frames for prediction and visualizes the most recent prediction on the video feed.

        Press 'q' or the 'Esc' key to exit the video processing loop.

        Note: This function uses OpenCV for video capture and display.

        No parameters or return value are explicitly defined in this function.
        '''
        cap = cv2.VideoCapture(0)
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            df_temp=self.add_bendshapes(frame)
            face_coords=self.add_coordinates(frame)
            if face_coords is not None and df_temp is not None :
                self.frame_queue.append(frame)
                self.dataframe=pd.concat([self.dataframe,df_temp],ignore_index=True)
                self.face_coordinates_queue.append(face_coords)
                if len(self.frame_queue) == 5:
                    self.predict_frames()
                    predicted_frame = self.frame_queue.popleft()
                    x, y, w, h = self.face_coordinates_queue.popleft()
                    cv2.rectangle(predicted_frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
                    cv2.putText(predicted_frame, self.label, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
                    cv2.imshow("Realtime Processing", predicted_frame)
                    self.dataframe = self.dataframe.drop(self.dataframe.index[0])
                    self.predictions.pop(0)
            elif face_coords is None or df_temp is None:
                cv2.imshow("Realtime Processing", frame)
                if self.dataframe.shape[0]>0:
                    self.dataframe = self.dataframe.drop(self.dataframe.index[0])
                    self.frame_queue.popleft()
                    self.face_coordinates_queue.popleft()
                    if len(self.predictions)>0:
                        self.predictions.pop(0)
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q') or key == 27:  
                break

        cap.release()
        cv2.destroyAllWindows()


model_path = "C:\\Users\\phuon\\Downloads\\trang_xinh_32.h5"
face_path=".vscode\\face_landmarker.task"
face_classifier = EmotionClassifier(model_path,face_path)
face_classifier.process_video()
