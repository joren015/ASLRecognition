import json
import uuid
import pandas as pd
import cv2
import torch
import matplotlib.pyplot as plt
import numpy as np
import mediapipe as mp
from os import walk, makedirs, listdir
from os.path import join, exists
from skimage.measure import compare_ssim
from skimage.feature import match_descriptors, plot_matches, CENSURE


def ResizeImgs(imgs, image_format):
    for load_path_i, save_path_i in imgs:
        img = cv2.imread(load_path_i)
        res = cv2.resize(img, dsize=(200, 200), interpolation=cv2.INTER_CUBIC)
        cv2.imwrite(save_path_i, res)
        torch.save(
            torch.from_numpy(res.reshape(3, 200, 200)),
            save_path_i.replace(image_format, ".pt"),
        )

    return True


def ChunkList(my_list, n):
    chunk_size = len(my_list) // n
    chunks = []
    for i in range(n - 1):
        ii = i * chunk_size
        chunks.append(my_list[ii : (ii + chunk_size)])

    last_chunk = (n - 1) * chunk_size
    chunks.append(my_list[last_chunk:-1])
    return chunks


def PreprocessDatasets(
    load_path, save_path, image_format, overwrite=False
):
    dataset = []
    imgs = []
    for root, dirs, files in walk(load_path):
        root_split = root.split("/")
        for i in range(len(files)):
            file_i = files[i]
            if file_i.endswith(image_format):
                load_path_i = "{}/{}".format(root, file_i)
                save_path_i = "{}/{}".format(save_path, file_i)
                if not exists(save_path_i) or overwrite:
                    save_path_base_i = "/".join(save_path_i.split("/")[0:-1])
                    if not exists(save_path_base_i):
                        makedirs(save_path_base_i)

                    imgs.append((load_path_i, save_path_i))

    print("Saving: {}".format(len(imgs)))
    ResizeImgs(imgs, image_format)


def MPAnnotate(file_list, save_dir, landmarks_save_dir, confidence=0.7):
    unclassified_data = []
    mp_drawing = mp.solutions.drawing_utils
    mp_hands = mp.solutions.hands
    hands = mp_hands.Hands(
        static_image_mode=True, max_num_hands=1, min_detection_confidence=confidence
    )
    for idx, file in enumerate(file_list):
        image = cv2.flip(cv2.imread(file), 1)
        results = hands.process(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
        if not results.multi_hand_landmarks:
            unclassified_data.append(file)
            continue

        annotated_image = image.copy()
        for hand_landmarks in results.multi_hand_landmarks:
            mp_drawing.draw_landmarks(
                annotated_image, hand_landmarks, mp_hands.HAND_CONNECTIONS
            )
            landmark_matrix = np.zeros((21, 3))
            i = 0
            for h in hand_landmarks.landmark:
                landmark_matrix[:][i] = np.array([h.x, h.y, h.z])
                i += 1

        filename_base = file.split("/")[-1].split(".")[-2]
        image_path = save_dir + filename_base
        landmarks_path = landmarks_save_dir + filename_base
        # cv2.imwrite(image_path + '_annotated.png', cv2.flip(annotated_image, 1))
        torch.save(torch.from_numpy(landmark_matrix), landmarks_path + "_landmarks.pt")

    hands.close()
    print(len(unclassified_data) / len(file_list))
    return unclassified_data


def MPLoop(file_list, save_dir, landmarks_save_dir):
    if not exists(save_dir):
        makedirs(save_dir)

    if not exists(landmarks_save_dir):
        makedirs(landmarks_save_dir)

    unclassified_data = MPAnnotate(file_list, save_dir, landmarks_save_dir, 0.99)
    confidence = 0.91
    while len(unclassified_data) > 0 and confidence > 0:
        confidence -= 0.1
        unclassified_data = MPAnnotate(
            unclassified_data, save_dir, landmarks_save_dir, confidence
        )

    print(
        "Unable to annotate {}% of data".format(
            100 * (len(unclassified_data) / len(file_list))
        )
    )
    print(len(unclassified_data) / len(file_list))


def GetPoints(im_path, pt_path):
    img = plt.imread(im_path)
    h, w, x = img.shape
    original_points = torch.load(pt_path)
    points = np.array(original_points)
    scaled_points = np.zeros([21, 2])
    scaled_points[:, 0] = points[:, 0] * w
    scaled_points[:, 1] = points[:, 1] * h
    flipped_points = scaled_points
    test = w / 2
    test2 = test - scaled_points[:, 0]
    test3 = test2 + test
    flipped_points[:, 0] = test3

    # plt.imshow(img)
    # for i in range(0,21,3):
    #   plt.plot(scaled_points[i,0], scaled_points[i,1], 'ro')
    #   plt.plot(scaled_points[i+1,0], scaled_points[i+1,1], 'go')
    #   plt.plot(scaled_points[i+2,0], scaled_points[i+2,1], 'bo')
    # plt.plot(scaled_points[2,0], scaled_points[2,1], 'ro')
    # plt.axis('off')
    # plt.show()

    return flipped_points


def HomographyTransform(img1, img2, fp1, fp2):
    M, mask = cv2.findHomography(fp1, fp2, cv2.RANSAC, 5.0)
    matchesMask = mask.ravel().tolist()
    h, w, d = img1.shape
    pts = np.float32([[0, 0], [0, h - 1], [w - 1, h - 1], [w - 1, 0]]).reshape(-1, 1, 2)
    dst = cv2.warpPerspective(img1, M, (img1.shape[1], img1.shape[0]))
    return dst


def CENSURETransform(img1, img2, fp1, fp2):
    bw_img1 = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY).astype("double")
    bw_img2 = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY).astype("double")
    cen = CENSURE()
    cen.detect(bw_img1)
    kp1 = cen.keypoints
    cen.detect(bw_img2)
    kp2 = cen.keypoints

    kp1 = np.append(kp1, fp1, axis=0)
    kp2 = np.append(kp2, fp2, axis=0)

    matches = match_descriptors(kp1, kp2, cross_check=True)

    # fig, ax = plt.subplots(nrows=1, ncols=1)
    # plot_matches(ax, img1, img2, kp1, kp2, matches)
    # plt.show()

    points1 = kp1[matches[:, 0]]
    points2 = kp2[matches[:, 1]]

    M, mask = cv2.findHomography(points1, points2, cv2.RANSAC)
    dst = cv2.warpPerspective(img1, M, (img1.shape[1], img1.shape[0]))

    return dst


def TransformEval(image_i, image_j, landmarks_i, landmarks_j, transform="homography"):
    try:
        img1 = plt.imread(image_i)
        img2 = plt.imread(image_j)
        fp1 = GetPoints(image_i, landmarks_i)
        fp2 = GetPoints(image_j, landmarks_j)
        if transform == "homography":
            dst = HomographyTransform(img1, img2, fp1, fp2)
        elif transform == "CENSURE":
            dst = CENSURETransform(img1, img2, fp1, fp2)
        else:
            raise Exception("Invalid transform")

        img1_gray = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)
        img2_gray = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY)
        dst_gray = cv2.cvtColor(dst, cv2.COLOR_BGR2GRAY)
        mssim1, grad1, S1 = compare_ssim(img1_gray, img2_gray, gradient=True, full=True)
        mssim2, grad2, S2 = compare_ssim(dst_gray, img2_gray, gradient=True, full=True)

        # f, axarr = plt.subplots(2,2)
        # axarr[0,0].imshow(img1)
        # axarr[0,1].imshow(img2)
        # axarr[1,0].imshow(dst)
        # axarr[1,1].imshow(dst)
        # plt.show()

        return mssim2 - mssim1
    except FileNotFoundError:
        print("File not found")
        return 0
    # except:
    #   print("Unexpected error:", sys.exc_info()[0])


def BaselineEval(image_paths, transform="homography"):
    i = 1
    results = {x: [] for x in image_paths}
    for image_i in image_paths:
        print("{}% complete".format(i / len(image_paths) * 100))
        scores = []
        landmarks_i = image_i.replace(".png", "_landmarks.pt")
        for image_j in image_paths:
            if not (image_j == image_i):
                landmarks_j = image_j.replace(".png", "_landmarks.pt")
                diff = TransformEval(
                    image_j, image_i, landmarks_i, landmarks_j, transform=transform
                )
                scores.append({image_j: diff})

        results[image_i] = scores
        # print("\n---------- Results ----------\n")
        # print(c)
        # print(image_i)
        # print("Samples: {}\nMean: {}\nMedian: {}\nMax: {}\nMin: {}".format(len(scores), sum(scores)/len(scores), median(scores), max(scores), min(scores)))
        i += 1

    return results


def DetermineNewBaseline(load_dir, save_dir):
  dirs = ["A", "B", "C", "D", "E", "F", "G", "H", "I", "J", "K", "L", "M", "N", "O", "P", "Q", "R", "S", "T", "U", "V", "W", "X", "Y", "Z"]
  results = {c: {} for c in dirs}
  transforms = ["homography"]
  for transform in transforms:
    for c in dirs:
      image_paths = []
      image_paths += ["{}/{}".format(load_dir, x) for x in listdir(load_dir) if x.split('.')[-1] == "png" and x.split("_")[2] == c]
      # print(image_paths)
      print("Character: {}".format(c))
      results[c] = BaselineEval(image_paths, transform)

    with open('{}/baseline_eval_{}_homemade_dataset.json'.format(save_dir, transform), 'w') as f:
      json.dump(results, f)


def NewBaseline(file):
    file1 = open(file)
    json_file = json.load(file1)

    dirs = [
        "A",
        "B",
        "C",
        "D",
        "E",
        "F",
        "G",
        "H",
        "I",
        "J",
        "K",
        "L",
        "M",
        "N",
        "O",
        "P",
        "Q",
        "R",
        "S",
        "T",
        "U",
        "V",
        "W",
        "X",
        "Y",
        "Z",
    ]
    baseline = ["" for x in range(len(dirs))]
    count1 = 0
    for c in dirs:  # A, B, C...
        json_layer1 = json_file[c]
        average = np.zeros([len(json_layer1)])
        ave_name = ["" for x in range(len(json_layer1))]
        count = 0
        for i in json_layer1:  # Image set as baseline
            json_layer2 = json_layer1[i]
            baseline_ave = np.zeros([len(json_layer2)])
            for j in range(len(json_layer2)):  # into 0,1,2...
                json_layer3 = json_layer2[j]
                for k in json_layer3:  # Test image, value
                    json_layer4 = json_layer3[k]
                    baseline_ave[j] = json_layer4
            ave_name[count] = i
            average[count] = np.mean(baseline_ave)
            count += 1
        maxval_index = np.argmax(average)
        baseline[count1] = ave_name[maxval_index]
        count1 += 1

    return baseline


def LabelDatasets(load_path, save_path, encoding_config):
    save_path_labels = "{}/labels.csv".format(save_path)
    save_path_encoding_dict = "{}/encoding.json".format(save_path)
    dataset = []
    encoding_dict = encoding_config

    if not exists(save_path):
        makedirs(save_path)

    for root, dirs, files in walk(load_path):
        root_split = root.split("/")
        for i in range(len(files)):
            file_i = files[i]
            if file_i.endswith(".pt") and "_landmarks" not in file_i:
                id = "{}_{}".format(root_split[-2], "{}".format(uuid.uuid4()))
                file_path = "{}/{}".format(root, file_i)
                label = file_i.split("_")[-2].lower()
                label_encoded = encoding_dict[label.lower()]
                dataset.append([id, file_path, label_encoded])

    df = pd.DataFrame(np.array(dataset), columns=["ID", "FilePath", "LabelEncoded"])
    df.to_csv(save_path_labels)
    with open(save_path_encoding_dict, "w") as f:
        json.dump(encoding_dict, f)
