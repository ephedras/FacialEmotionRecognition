from facial_fer_model import FacialExpressionRecog
import datetime
import cv2 as cv
import numpy as np

def visualize(image, det_res, fer_res, box_color=(0, 255, 0), text_color=(0, 0, 255)):
    fer_type =''
    print('%s %3d faces detected.' % (datetime.datetime.now(), len(det_res)))

    output = image.copy()
    landmark_color = [
        (255,  0,   0),  # right eye
        (0,    0, 255),  # left eye
        (0,  255,   0),  # nose tip
        (255,  0, 255),  # right mouth corner
        (0,  255, 255)   # left mouth corner
    ]

    for ind, (det, fer_type) in enumerate(zip(det_res, fer_res)):
        bbox = det[0:4].astype(np.int32)
        fer_type = FacialExpressionRecog.getDesc(fer_type)
        print("Face %2d: %d %d %d %d %s." % (ind, bbox[0], bbox[1], bbox[0]+bbox[2], bbox[1]+bbox[3], fer_type))
        cv.rectangle(output, (bbox[0], bbox[1]), (bbox[0]+bbox[2], bbox[1]+bbox[3]), box_color, 2)
        cv.putText(output, fer_type, (bbox[0], bbox[1]+12), cv.FONT_HERSHEY_DUPLEX, 0.5, text_color)
        
        # expression_counts[fer_type]+=1
        landmarks = det[4:14].astype(np.int32).reshape((5, 2))
        for idx, landmark in enumerate(landmarks):
            cv.circle(output, landmark, 2, landmark_color[idx], 2)
    return output,fer_type


def process(detect_model, fer_model, frame):
    h, w, _ = frame.shape
    detect_model.setInputSize([w, h])
    dets = detect_model.infer(frame)

    if dets is None:
        return False, None, None

    fer_res = np.zeros(0, dtype=np.int8)
    for face_points in dets:
        fer_res = np.concatenate((fer_res, fer_model.infer(frame, face_points[:-1])), axis=0)
    return True, dets, fer_res
