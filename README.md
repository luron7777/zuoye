第一步：环境的配置
在按照老师教的步骤进行配置的过程中出现下述错误：
(gradio)Ps c:\users\10920\Desktop\FaceProcessingWebcam\FaceAnalysisWebApp> python main.py
Error in sys.excepthook:
Original exception was:
(gradio)Ps c:\users\10920\Desktop\FaceProcessingWebcam\FaceAnalysiswebApp>
看了一下源代码，感觉可能是运用了老师说的4.39.0版本的gradio，所以在requirment.txt文件中将gradio版本指定为4.39之后发现在本地网址运行时，在点了拍照后出现连接错误的提示如下：
pydantic.errors.PydanticSchemaGenerationError: Unable to generate pydantic-core schema for <class 'starlette.requests.Request'>. Set `arbitrary_types_allowed=True` in the model_config to ignore this error or implement `__get_pydantic_core_schema__` on your type to fully support it.

If you got this error by calling handler(<some type>) within `__get_pydantic_core_schema__` then you likely need to call `handler.generate_schema(<some type>)` since we do not call `__get_pydantic_core_schema__` on `<some type>` otherwise to avoid infinite recursion.

For further information visit https://errors.pydantic.dev/2.9/u/schema-for-unknown-type.


在网上找了圈提示之后，将gradio改成3.39，将gradio_client改成0.17.0,将mediapipe指定为0.10.10重新按照requirment.txt文件进行配置之后，会发现本地连接拍照能用了，但是视频连接还会出现错误。错误是ffmpy文件中没有ffmpeg这一项，于是上网找了一圈，就按照链接：https://blog.csdn.net/u011027547/article/details/122490254?ops_request_misc=%257B%2522request%255Fid%2522%253A%2522A1ADC4E9-9106-4A4E-9540-610A1908B970%2522%252C%2522scm%2522%253A%252220140713.130102334..%2522%257D&request_id=A1ADC4E9-9106-4A4E-9540-610A1908B970&biz_id=0&utm_medium=distribute.pc_search_result.none-task-blog-2~all~sobaiduend~default-1-122490254-null-null.142^v100^pc_search_result_base5&utm_term=pycharm%E4%B8%ADffmpeg%E5%BA%93%E5%AE%89%E8%A3%85&spm=1018.2226.3001.4187
一步步的安装ffmpeg库，然后配置好环境变量后，就发现能够正常使用视频和在线流了。

第二步：是将校徽放到人脸上:
参考https://docs.opencv.org/4.x/dc/d2c/tutorial_real_time_pose.html 链接将代码写好后，出现如下问题：Traceback (most recent call last):
  File "C:\Users\10920\Desktop\2\FaceProcessingWebcam\FaceAnalysisWebApp\1.py", line 41, in <module>
    mask = (warped_logo[..., 3] > 0).astype(np.uint8)  # alpha通道作为掩码
IndexError: index 3 is out of bounds for axis 2 with size 3
查了原因，发现是因为加载的校徽图像没有 alpha 通道。通常，PNG 图像会包含透明度通道（alpha 通道），但如果图像格式不正确，可能只会有 RGB 三个通道，而我的格式正好是.jpg，将图片格式换掉后能够正常运行。但是会发现由于校徽图片过大，导致整个人脸照片都被校徽给挡住了，具体步骤代码如下：
if results.detections:
            for detection in results.detections:
                bboxC = detection.location_data.relative_bounding_box
                h, w, _ = frame.shape
                
                # 计算人脸位置
                x1, y1, width, height = int(bboxC.xmin * w), int(bboxC.ymin * h), \
                                        int(bboxC.width * w), int(bboxC.height * h)

                # 设置校徽位置在头部上方
                logo_h, logo_w, _ = logo.shape
                logo_y1 = max(y1 - logo_h // 2, 0)  # 确保不超出边界
                logo_x1 = x1 + width // 2 - logo_w // 2  # 居中显示
                logo_y2 = logo_y1 + logo_h
                logo_x2 = logo_x1 + logo_w
                
                # 获取透视变换矩阵
                src_pts = np.float32([[0, 0], [logo_w, 0], [logo_w, logo_h], [0, logo_h]])
                dst_pts = np.float32([[logo_x1, logo_y1], 
                                       [logo_x2, logo_y1], 
                                       [logo_x2, logo_y2], 
                                       [logo_x1, logo_y2]])

                matrix = cv2.getPerspectiveTransform(src_pts, dst_pts)
                warped_logo = cv2.warpPerspective(logo, matrix, (w, h))

                # 创建掩膜
                mask = warped_logo[..., 3]  # alpha 通道
                mask_inv = cv2.bitwise_not(mask)
                logo_rgb = warped_logo[..., :3]

重新看了几遍代码之后，在查询opencv中Haar级联检测人脸的用法后，改成下面代码（一部分）：
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
    修改完后发现能够正常运行，能将校徽放到人脸正中间，并且可以根据人脸的大小进行进行放缩，并使校徽放在人脸中间


    第三步：调用增加新的facial landmark detection模型
    首先根据源代码保留了点击拍照功能，然后在参考链接：https://blog.csdn.net/hao_san_520/article/details/82081513?utm_medium=distribute.pc_relevant.none-task-blog-2~default~baidujs_baidulandingword~default-4-82081513-blog-86165960.235^v43^pc_blog_bottom_relevance_base2&spm=1001.2101.3001.4242.3&utm_relevant_index=7
    代码之后，进行修改改成我的add_Mediapipe函数部分，单独运行时发现他识别的时候只有一点，并没有68个，在找了一圈资料之后，将代码部分改成：
     hui = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    Frames = detect(hui)
    for img in Frames:
        landmarks = predict(hui, img)
        for num in range(68):
            x, y = landmarks.part(num).x, landmarks.part(num).y
            cv2.circle(image, (x, y), 3, (0, 55, 55), -1)

    之后，发现能够准确的进行识别并画出了68个点
    在根据https://github.com/DefTruth/torchlm 链接下的68个点的代码例子，写入：
import torchlm
from torchlm.torchlm.tools import faceboxesv2
from torchlm.torchlm.models import pipnet as an
（因为我的torchlm库放在torchlm包里边所以多一层torchlm,所以导致后面代码中我的torchlm.runtime直接这么使用之后，发现我的Torchlm，和add_Mediapipe_Torchlm部分出现链接错误，在在仔细浏览全部代码时，才正确的改过来）

写出torchlm的检测代码如下：
  torchlm.torchlm.runtime.bind(faceboxesv2(device="cuda"))
    torchlm.torchlm.runtime.bind(an("pipnet_resnet18_10x68x32x256_300w", backbone="resnet18"))
    landmarks, _ = torchlm.torchlm.runtime.forward(image)
    image_with_landmarks = image.copy()
    for landmark in landmarks:
        for (x, y) in landmark:
            cv2.circle(image_with_landmarks, (int(x), int(y)), radius=3, color=(255, 0, 0), thickness=-1)
    return image_with_landmarks
单独运行发现能够正常使用。
然后再利用line函数将两个部分的点连接起来得到add_Mediapipe_Torchlm函数
之后就是配置ui窗口，将ui改成我想要的样子。

