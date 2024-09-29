import cv2
import gradio as gr
import mediapipe as mp
import dlib
import torchlm
from torchlm.torchlm.tools import faceboxesv2
from torchlm.torchlm.models import pipnet as an

mp_drawing = mp.solutions.drawing_utils
mp_face_mesh = mp.solutions.face_mesh
detect = dlib.get_frontal_face_detector()
predict = dlib.shape_predictor("model/shape_predictor_68_face_landmarks.dat")

def Mediapipe(image):
    hui = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    Frames = detect(hui)
    for img in Frames:
        landmarks = predict(hui, img)
        for num in range(68):
            x, y = landmarks.part(num).x, landmarks.part(num).y
            cv2.circle(image, (x, y), 3, (0, 55, 55), -1)
    return image

def Torchlm(image):
    torchlm.torchlm.runtime.bind(faceboxesv2(device="cuda"))
    torchlm.torchlm.runtime.bind(an("pipnet_resnet18_10x68x32x256_300w", backbone="resnet18"))
    landmarks, _ = torchlm.torchlm.runtime.forward(image)
    image_with_landmarks = image.copy()
    for landmark in landmarks:
        for (x, y) in landmark:
            cv2.circle(image_with_landmarks, (int(x), int(y)), radius=3, color=(255, 0, 0), thickness=-1)
    return image_with_landmarks

def add_Mediapipe_Torchlm(image):
    torchlm.torchlm.runtime.bind(faceboxesv2(device="cuda"))
    torchlm.torchlm.runtime.bind(an("pipnet_resnet18_10x68x32x256_300w", backbone="resnet18"))
    landmarks, _ = torchlm.torchlm.runtime.forward(image)
    landmark1 = [
        (x, y) for landmark in landmarks
        for (x, y) in landmark
    ]
    hui = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    Frames = detect(hui)

    for img in Frames:
        landmark2 = predict(hui, img)
        dlib_landmarks = [(landmark2.part(i).x, landmark2.part(i).y) for i in range(68)]
        comparison = image.copy()
        for pt1, pt2 in zip(landmark1, dlib_landmarks):
            cv2.line(comparison, tuple(map(int, pt1)), pt2, (66, 66, 66), 4)
        return comparison

class FaceProcessing:
    def __init__(self, ui_obj):
        self.ui_obj = ui_obj

    def take_webcam_photo(self, image):
        return image

    def add_Mediapipe(self, image):
        return Mediapipe(image)

    def add_Torchlm(self, image):
        return Torchlm(image)

    def add_Mediapipe_Torchlm(self, image):
        return add_Mediapipe_Torchlm(image)

    def create_ui(self):
        with self.ui_obj:
            gr.Markdown("Face Analysis with Webcam/Video")
            with gr.Tabs():
                with gr.TabItem('Models Comparison'):
                    with gr.Row():
                        with gr.Column():
                            comparison_image_in = gr.Image(label="Webcam Image Input", source="webcam")
                        with gr.Column():
                            comparison_photo_action = gr.Button("Take the Photo")
                            comparison_add_Mediapipe_action = gr.Button("Apply add Mediapipe 68 points ")
                            comparison_add_Torchlm_action = gr.Button("Apply add Torchlm 68 points")
                            comparison_add_Mediapipe_Torchlm_action = gr.Button("Apply add Mediapipe Torchlm ")

                    with gr.Row():
                        comparison_image_out = gr.Image(label="Webcam Image Output")
                        comparison_mediapipe_out = gr.Image(label="Webcam Mediapipe 68 points Output")
                        comparison_torchlm_out = gr.Image(label="Webcam Torchlm 68 points Output")
                        comparison_mediapipe_add_torchlm_out = gr.Image(label="Webcam connect two model Output")


            comparison_photo_action.click(
                self.take_webcam_photo,
                [
                    comparison_image_in
                ],
                [
                    comparison_image_out
                ]
            )
            comparison_add_Mediapipe_action.click(
                self.add_Mediapipe,
                [
                    comparison_image_in
                ],
                [
                    comparison_mediapipe_out
                ]
            )
            comparison_add_Torchlm_action.click(
                self.add_Torchlm,
                [
                    comparison_image_in
                ],
                [
                    comparison_torchlm_out
                ]
            )
            comparison_add_Mediapipe_Torchlm_action.click(
                self.add_Mediapipe_Torchlm,
                [
                    comparison_image_in
                ],
                [
                    comparison_mediapipe_add_torchlm_out
                ]
            )

    def launch_ui(self):
        self.ui_obj.launch()

if __name__ == '__main__':
    my_app = gr.Blocks()
    face_ui = FaceProcessing(my_app)
    face_ui.create_ui()
    face_ui.launch_ui()
