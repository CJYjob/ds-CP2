
import os
import numpy as np

def calc_distance(feat1, feat2):
    distance = np.linalg.norm(feat1 - feat2)
    return distance

if __name__ == "__main__":

    import cv2
    import settings
    from reid_onnx_helper import ReidHelper
    
    helper = ReidHelper(settings.ReID)

    BASE_FOLDER = "sample"

    feat1 = None
    min_distance = 0
    min_file = None

    files = os.listdir(BASE_FOLDER)
    for index, file in enumerate(files):
        file_path = os.path.join(BASE_FOLDER, file)
        car_img = cv2.imread(file_path)

        feat = helper.infer(car_img)
        if index == 0:
            feat1 = feat
        else:
            distance = calc_distance(feat1, feat)
            print(file, distance)

            if min_distance == 0 or min_distance > distance:
                min_distance = distance
                min_file = file

    print("target_file:", files[0])
    print("min_file:", min_file)



