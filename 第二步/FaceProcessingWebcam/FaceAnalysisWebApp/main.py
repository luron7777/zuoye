import cv2
import numpy as np
import gradio as gr

def overlay_logo_on_face(face_image, logo_image):
    # 转换为灰度图像
    gray_face = cv2.cvtColor(face_image, cv2.COLOR_BGR2GRAY)

    # 使用 Haar 级联检测人脸
    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
    faces = face_cascade.detectMultiScale(gray_face, scaleFactor=1.1, minNeighbors=5)

    for (x, y, w, h) in faces:
        # 计算人脸中心
        center_x, center_y = x + w // 2, y + h // 2

        # 缩放校徽图像到人脸的大小
        logo_resized = cv2.resize(logo_image, (w // 3, h // 3))  # 调整比例

        # 计算校徽的位置
        logo_height, logo_width = logo_resized.shape[:2]
        start_x = center_x - logo_width // 2
        start_y = center_y - logo_height // 2

        # 确保不会超出边界
        if start_x < 0: start_x = 0
        if start_y < 0: start_y = 0
        if start_x + logo_width > face_image.shape[1]:
            start_x = face_image.shape[1] - logo_width
        if start_y + logo_height > face_image.shape[0]:
            start_y = face_image.shape[0] - logo_height

        # 创建掩码并合成校徽到人脸图像上
        for c in range(0, 3):
            face_image[start_y:start_y + logo_height, start_x:start_x + logo_width, c] = \
                logo_resized[:, :, c] * (logo_resized[:, :, 2] / 255.0) + \
                face_image[start_y:start_y + logo_height, start_x:start_x + logo_width, c] * (1 - logo_resized[:, :, 2] / 255.0)

    return face_image

# Gradio 接口
def process_image(face_image, logo_image):
    return overlay_logo_on_face(face_image, logo_image)

# 加载你的校徽图像
logo_image = cv2.imread(r'C:\Users\10920\Desktop\2\FaceProcessingWebcam\FaceAnalysisWebApp\xiaohui.png', cv2.IMREAD_UNCHANGED)

iface = gr.Interface(fn=process_image,
                     inputs=[gr.Image(type="numpy", label="Upload Face Image"),
                             gr.Image(type="numpy", label="Upload Logo Image")],
                     outputs=gr.Image(type="numpy", label="Output Image"))

iface.launch()