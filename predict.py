import os
import re
import sys
import time
import logging
import argparse
from pathlib import Path


import cv2
import torch
import numpy as np
from PIL import Image
from albumentations.augmentations.functional import image_compression
from torchvision.transforms import Normalize

from training.zoo.classifiers import DeepFakeClassifier
from preprocessing.retinaface.detect import FaceDetector

stdout_handler = logging.StreamHandler(sys.stdout)
stdout_handler.setLevel(logging.INFO)
handlers = [stdout_handler]
logging.basicConfig(
    level=logging.INFO,
    format="%(levelname)s; %(asctime)s; %(module)s:%(funcName)s:%(lineno)d; %(message)s",
    handlers=handlers)

logger = logging.getLogger(__name__)


def read_models(model_paths):
    models = []
    for path in model_paths:
        device = "cuda" if torch.cuda.is_available() else "cpu"
        model = DeepFakeClassifier(encoder="tf_efficientnet_b7_ns").to(device)
        logger.info("loading state dict {}".format(path))
        checkpoint = torch.load(path, map_location="cpu")
        state_dict = checkpoint.get("state_dict", checkpoint)
        model.load_state_dict(
            {re.sub("^module.", "", k): v.to(torch.float32) for k, v in state_dict.items()}, strict=True)
        model.eval()
        del checkpoint
        models.append(model)
    return models


def put_to_center(img, input_size):
    img = img[:input_size, :input_size]
    image = np.zeros((input_size, input_size, 3), dtype=np.uint8)
    start_w = (input_size - img.shape[1]) // 2
    start_h = (input_size - img.shape[0]) // 2
    image[start_h:start_h + img.shape[0],
          start_w: start_w + img.shape[1], :] = img
    return image


def isotropically_resize_image(img, size, interpolation_down=cv2.INTER_AREA, interpolation_up=cv2.INTER_CUBIC):
    h, w = img.shape[:2]
    if max(w, h) == size:
        return img
    if w > h:
        scale = size / w
        h = h * scale
        w = size
    else:
        scale = size / h
        w = w * scale
        h = size
    interpolation = interpolation_up if scale > 1 else interpolation_down
    resized = cv2.resize(img, (int(w), int(h)), interpolation=interpolation)
    return resized


if __name__ == '__main__':
    parser = argparse.ArgumentParser("Predict test videos")
    arg = parser.add_argument
    arg('--weights-dir', type=str, default="weights",
        help="path to directory with checkpoints")
    arg('--models', nargs='+', required=True, help="checkpoint files")
    args = parser.parse_args()

    if not args.models and not Path(args.weights_dir).exists():
        raise ValueError(
            "You need to specify where are the weights and model names")

    model_paths = [os.path.join(args.weights_dir, model)
                   for model in args.models]
    models = read_models(model_paths)
    num_frames = 1

    detector = FaceDetector(
        network="mobile0.25", weights="./weights/retinaface/mobilenet0.25_Final.pth")

    input_size = 380
    device = "cuda" if torch.cuda.is_available() else "cpu"
    mean = [0.485, 0.456, 0.406]
    std = [0.229, 0.224, 0.225]
    normalize_transform = Normalize(mean, std)

    while True:
        frames = []
        capture = cv2.VideoCapture(0)
        for i in range(num_frames):
            _, frame = capture.read()
            frames.append(frame)
        frames = np.stack(arrays=frames, axis=0)
        results = []
        for i, frame in enumerate(frames):
            h, w = frame.shape[:2]
            img = Image.fromarray(frame.astype(np.uint8))
            img = img.resize(size=[s // 2 for s in img.size])

            img_array = np.array(img)
            detect_start = time.time()
            _, drawed_img, aligned = detector.detect(
                np.array(img, dtype=np.float32), landmarks=True)
            # cv2.imshow(f"webcam", drawed_img)
            frame = aligned
            annotations, _, _ = detector.detect(
                np.array(frame, dtype=np.float32), landmarks=True)
            cv2.imwrite("to_predict.jpg", frame)
            print(
                f"Time usage of detecting: {time.time() - detect_start}s")
            batch_boxes = annotations["bbox"]
            batch_boxes = [np.reshape(np.array(box, dtype=np.float32), (4, ))
                           for box in batch_boxes]
            faces = []
            if batch_boxes is None or len(batch_boxes) == 0:
                continue
            for bbox in batch_boxes:
                if bbox is not None:
                    xmin, ymin, xmax, ymax = [int(b) for b in bbox]
                    w = xmax - xmin
                    h = ymax - ymin
                    additional_h = h // 3
                    additional_w = w // 3
                    crop = frame[max(ymin - additional_h, 0):ymax +
                                 additional_h, max(xmin - additional_w, 0):xmax + additional_w]

                    cv2.imwrite(f"crop.jpg", crop)
                    faces.append(crop)

            frame_dict = {
                "frame_w": w,
                "frame_h": h,
                "faces": faces,
            }
            results.append(frame_dict)

        for result in results:
            faces = result.get("faces", "")
            if len(faces) == 0:
                continue

            resized_faces = []
            for face in faces:
                resized_face = isotropically_resize_image(face, input_size)
                cv2.imwrite(f"resize.jpg", resized_face)
                resized_face = put_to_center(resized_face, input_size)
                cv2.imwrite(f"center.jpg", resized_face)
                resized_faces.append(resized_face)
                resized_faces.append(resized_face)

                # if apply_compression:
                #     resized_face = image_compression(
                #         resized_face, quality=90, image_type=".jpg")

            x = np.stack(resized_faces)
            x = torch.tensor(x, device=device).float()

            # Preprocess the images.
            x = x.permute((0, 3, 1, 2))
            for i in range(len(x)):
                x[i] = normalize_transform(x[i] / 255.)

            with torch.no_grad():
                predict_start = time.time()
                models_result = []
                for model in models:
                    faces_pred = model(x)
                    faces_pred = torch.sigmoid(faces_pred.squeeze())
                    faces_result = []
                    for face in faces_pred:
                        faces_result.append(face.item())
                    models_result.append(np.mean(faces_result))
                frame_confid = np.mean(models_result)

            print(
                f"time usage of predicting: {time.time() - predict_start}s")

            cv2.putText(drawed_img, f"confidence to be fake: {str(round(frame_confid, 2))}", (10, 40),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 1, cv2.LINE_AA)
            cv2.imshow(f"webcam", drawed_img)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

    capture.release()
