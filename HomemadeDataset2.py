from NotebookUtils import PrintDatetime
PrintDatetime("Importing packages")
import itertools  
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import mediapipe as mp
import shutil
from os import walk, makedirs, listdir, cpu_count, environ
from os.path import join, exists, isfile, join
from sklearn.model_selection import train_test_split
from Functions import *
from Model import *
PrintDatetime()


root_dir = "/mnt/d/School/Masters/CSCI5561ComputerVision/Project/content"
dataset_path = "{}/Datasets".format(root_dir)
if not exists(root_dir):
  makedirs(root_dir)

try:
  shutil.rmtree(root_dir + "/Datasets/silver")
  shutil.rmtree(root_dir + "/Datasets/gold")
except OSError as e:
  print("Error: %s : %s" % (root_dir, e.strerror))

directories = ["{}/models", "{}/Datasets/bronze/homemade_dataset2", "{}/Datasets/silver/homemade_dataset2", "{}/Datasets/silver/homemade_dataset2_cropped/center_2", "{}/Datasets/silver/homemade_dataset2_resized/center_2", "{}/Datasets/gold/Homography", "{}/Datasets/gold/Censure", "{}/Datasets/gold/Transforms", "{}/Datasets/gold/homemade_dataset2_resized/center_2", "{}/Datasets/gold/homemade_dataset2_transformed/center_2/"]
directories = [x.format(root_dir) for x in directories]
for d in directories:
  if not exists(d):
    makedirs(d)


starting_data = {"./Datasets/HomemadeDataset2/center_2.mp4": "{}/bronze/homemade_dataset2/center_2.mp4".format(dataset_path), "./Datasets/HomemadeDataset2/center_2.csv": "{}/bronze/homemade_dataset2/center_2.csv".format(dataset_path)}
for k,v in starting_data.items():
  if not exists(v):
    shutil.copyfile(k, v)




train_test_path_base = "{}/gold".format(dataset_path)
alpha_num = ["0", "1", "2", "3", "4", "5", "6", "7", "8", "9", "a", "b", "c", "d", "e", "f", "g", "h", "i", "j", "k", "l", "m", "n", "o", "p", "q", "r", "s", "t", "u", "v", "w", "x", "y", "z"]
encoding_config = {"a": 0, "b": 1,"c": 2,"d": 3,"e": 4,"f": 5,"g": 6,"h": 7
                   ,"i": 8,"j": 9,"k": 10,"l": 11,"m": 12,"n": 13,"o": 14
                   ,"p": 15,"q": 16,"r": 17,"s": 18,"t": 19,"u": 20,"v": 21
                   ,"w": 22,"x": 23,"y": 24,"z": 25,"0": 26,"1": 27,"2": 28
                   ,"3": 29,"4": 30,"5": 31,"6": 32,"7": 33,"8": 34,"9": 35 }
PrintDatetime()



PrintDatetime("Started extracting video dataset")
video_dir_load = dataset_path + "/bronze/homemade_dataset2"
labels_dir_load = dataset_path + "/bronze/homemade_dataset2"
file_list = ["{}/{}".format(video_dir_load, x) for x in listdir(video_dir_load) if '.mp4' in x and 'center' in x]
mp_drawing = mp.solutions.drawing_utils
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(min_detection_confidence=0.7, max_num_hands=1, static_image_mode=False, min_tracking_confidence=0.5)
for f in file_list:
  print(f)
  file_name = f.split('/')[-1]
  file_name_no_ext = file_name.split('.')[0]
  output_file_path = '/'.join(f.split('/')[0:-1] + [file_name_no_ext]).replace("bronze", "silver")
  if not exists(output_file_path):
    makedirs(output_file_path)

  labels_path = "{}/{}.csv".format(labels_dir_load, file_name_no_ext)
  labels = pd.read_csv(labels_path)
  rotate_video = False
  if "center" in file_name:
    rotate_video = True

  vid = cv2.VideoCapture(f)
  count_frame = 0
  label_i = '-'
  while(vid.isOpened()):
    ret, frame = vid.read()
    if ret:
      subset = labels[(labels['frame_start'] <= count_frame) & (labels['frame_end'] > count_frame)]
      if len(subset) == 1:
        label_i = subset.iloc[0]['label']
      elif len(subset) == 0:
        label_i = '-'
      else:
        print("SOMETHING WENT WRONG")

      if rotate_video:
        frame = cv2.rotate(frame, cv2.ROTATE_180)

      if label_i != "-":
        export_file = "{}/{}_{}_{}".format(output_file_path, file_name_no_ext, label_i, count_frame)
        cv2.imwrite(export_file + ".png", frame)
        image = cv2.cvtColor(cv2.flip(frame, 1), cv2.COLOR_BGR2RGB)
        image.flags.writeable = False
        results = hands.process(image)
        image.flags.writeable = True
        image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
        if results.multi_hand_landmarks:
          for hand_landmarks in results.multi_hand_landmarks:
            mp_drawing.draw_landmarks(image, hand_landmarks, mp_hands.HAND_CONNECTIONS)
            landmark_matrix = np.zeros((21,3))
            i = 0
            for h in hand_landmarks.landmark:
              landmark_matrix[:][i] = np.array([h.x, h.y, h.z])
              i += 1

            torch.save(torch.from_numpy(landmark_matrix), export_file + "_landmarks.pt")
        
        # cv2.imshow('MediaPipe Hands', image)
      
      count_frame += 1
      if cv2.waitKey(1) & 0xFF == ord('q'):
        break

    else:
      break

  vid.release()
  cv2.destroyAllWindows()



PrintDatetime("Started cropping video dataset")
letters = ["A", "B", "C", "D", "E", "F", "G", "H", "I", "J", "K", "L", "M", "N", "O", "P", "Q", "R", "S", "T", "U", "V", "W", "X", "Y", "Z"]
dirs = ["center_2"]
for i in dirs:
  image_dir = "{}/silver/homemade_dataset2/{}/".format(dataset_path, i)
  file_list = ["{}{}".format(image_dir, x) for x in listdir(image_dir) if ".png" in x]
  save_dir = "{}/silver/homemade_dataset2_cropped/{}".format(dataset_path, i)
  for l in letters:
    sub_list = [x for x in file_list if x.split('/')[-1].split('_')[2] == l]

    for idx, file in enumerate(sub_list):
      file_name = file.split('/')[-1]
      landmarks_path = '/'.join(file.split('.')[0:-1]) + "_landmarks.pt"
      if not exists(landmarks_path):
        print("No landmarks")
        MPLoop([file], "{}/misc/".format(root_dir), file.replace(file_name, ""))
        points = GetPoints(file, landmarks_path)
      else:
        points = GetPoints(file, landmarks_path)
      
      min_x = int(np.min(points[:,0]))-50
      max_x = int(np.max(points[:,0]))+50
      min_y = int(np.min(points[:,1]))-100
      max_y = int(np.max(points[:,1]))+100
      img = plt.imread(file)
      img_cropped = img[min_y:max_y,min_x:max_x,:]
      h,w,x = img_cropped.shape
      points[:,0] = points[:,0] - min_x
      points[:,1] = points[:,1] - min_y
      reverted_points = np.zeros([21,3])
      reverted_points[:,0] = (w/2 - points[:,0] + w/2)/w
      reverted_points[:,1] = points[:,1]/h
      torch.save(torch.from_numpy(reverted_points), landmarks_path.replace("silver/homemade_dataset2", "gold/homemade_dataset2_resized"))
      plt.imsave("{}/{}".format(save_dir, file_name), img_cropped)
      # plt.imshow(img_cropped)
      # plt.show()

PrintDatetime()


PrintDatetime("Started resizing video dataset")
PreprocessDatasets("{}/silver/homemade_dataset2_cropped/center_2".format(dataset_path), "{}/gold/homemade_dataset2_resized/center_2".format(dataset_path), ".png", overwrite=True)
PrintDatetime()



PrintDatetime("Started determining new baseline")
baseline_results_path = "{}/bronze/homemade_dataset2".format(dataset_path)
DetermineNewBaseline("{}/gold/homemade_dataset2_resized/center_2".format(dataset_path), baseline_results_path, is_still=False)
PrintDatetime()



PrintDatetime("Started transforming video dataset")
baseline2 = NewBaseline(baseline_results_path + "/baseline_eval_homography_homemade_dataset.json", is_still=False)
dirs = ["A", "B", "C", "D", "E", "F", "G", "H", "I", "J", "K", "L", "M", "N", "O", "P", "Q", "R", "S", "T", "U", "V", "W", "X", "Y", "Z"]
results = {c: {} for c in dirs}
transform = "homography"
for c in dirs:
  image_paths = []
  image_dir = "{}/gold/homemade_dataset2_resized/center_2".format(dataset_path)
  image_paths += ["{}/{}".format(image_dir, x) for x in listdir(image_dir) if x.split('.')[-1] == "png" and x.split("_")[2] == c]
  baseline_img_path = [x for x in baseline2 if x.split('/')[-1].split('_')[2] == c][0]
  baseline_fp_path = baseline_img_path.replace(".png", "_landmarks.pt")
  baseline_img = cv2.imread(baseline_img_path)
  baseline_fp = GetPoints(baseline_img_path, baseline_fp_path)
  for img_path in image_paths:
    if img_path != baseline_img_path:
      img_fp_path = img_path.replace(".png", "_landmarks.pt")
      img = cv2.imread(img_path)
      img_fp = GetPoints(img_path, img_fp_path)
      img_transformed = HomographyTransform(img, baseline_img, img_fp, baseline_fp)
      cv2.imwrite(img_path.replace("homemade_dataset2_resized", "homemade_dataset2_transformed"), img_transformed)
      torch.save(torch.from_numpy(img_transformed.reshape(3, 200, 200)), img_path.replace("homemade_dataset2_resized", "homemade_dataset2_transformed").replace(".png", ".pt"))
      

  print("Character: {}".format(c))

PrintDatetime()



PrintDatetime("Started labeling non-transformed video data")
LabelDatasets(load_path="{}/gold/homemade_dataset2_resized".format(dataset_path), save_path="{}/gold/homemade_dataset2_resized".format(dataset_path), encoding_config=encoding_config)
PrintDatetime()



PrintDatetime("Started labeling transformed video data")
LabelDatasets(load_path="{}/gold/homemade_dataset2_transformed".format(dataset_path), save_path="{}/gold/homemade_dataset2_transformed".format(dataset_path), encoding_config=encoding_config)
PrintDatetime()



PrintDatetime("Started training")
# hpt = {"learning_rate": [1e-2, 1e-4, 1e-5], "batch_size": [8, 32, 128]}
hpt = {"learning_rate": [1e-2], "batch_size": [32]}
configs = [1, 2]
keys, values = zip(*hpt.items())
i = 1
for config in configs:
  for v in itertools.product(*values):
    model_save_path_base = "{}/models/homemade_dataset2/Base/{}".format(root_dir, i)
    model_save_path_transformed = "{}/models/homemade_dataset2/Transformed/{}".format(root_dir, i)
    makedirs(model_save_path_base)
    makedirs(model_save_path_transformed)
    experiment = dict(zip(keys, v))
    print(experiment)
    learning_rate = experiment["learning_rate"]
    batch_size = experiment["batch_size"]

    PrintDatetime("Started")
    homemade_train_test_path = "{}/homemade_dataset2_resized".format(train_test_path_base)
    full_dataset = pd.read_csv("{}/labels.csv".format(homemade_train_test_path))
    az_dataset = full_dataset[full_dataset["LabelEncoded"] <= 25]
    train_dataset, test_val_dataset = train_test_split(az_dataset, test_size=0.2, random_state=0, shuffle=True)
    test_dataset, val_dataset = train_test_split(test_val_dataset, test_size=0.5, random_state=0, shuffle=True)
    train_dataset.to_csv("{}/train_az.csv".format(homemade_train_test_path))
    test_dataset.to_csv("{}/test_az.csv".format(homemade_train_test_path))
    val_dataset.to_csv("{}/val_az.csv".format(homemade_train_test_path))
    main("{}/train_az.csv".format(homemade_train_test_path), "{}/test_az.csv".format(homemade_train_test_path), "{}/val_az.csv".format(homemade_train_test_path), learning_rate=learning_rate, batch_size=batch_size, num_epochs=10, model_save_path=model_save_path_base)
    PrintDatetime()

    homemade_train_test_path_transformed = "{}/homemade_dataset2_transformed".format(train_test_path_base)
    full_dataset = pd.read_csv("{}/labels.csv".format(homemade_train_test_path_transformed))
    az_dataset = full_dataset[full_dataset["LabelEncoded"] <= 25]
    train_dataset, test_val_dataset = train_test_split(az_dataset, test_size=0.2, random_state=0, shuffle=True)
    test_dataset, val_dataset = train_test_split(test_val_dataset, test_size=0.5, random_state=0, shuffle=True)
    train_dataset.to_csv("{}/train_az.csv".format(homemade_train_test_path_transformed))
    if config == 1:
      config_train_path = "{}/train_az.csv".format(homemade_train_test_path_transformed)
    elif config == 2:
      config_train_path = "{}/train_az.csv".format(homemade_train_test_path)

    test_dataset.to_csv("{}/test_az.csv".format(homemade_train_test_path_transformed))
    val_dataset.to_csv("{}/val_az.csv".format(homemade_train_test_path_transformed))
    PrintDatetime("Started training experiment")
    main(config_train_path, "{}/test_az.csv".format(homemade_train_test_path_transformed), "{}/val_az.csv".format(homemade_train_test_path_transformed), learning_rate=learning_rate, batch_size=batch_size, num_epochs=10, model_save_path=model_save_path_transformed)
    PrintDatetime()
    i += 1

PrintDatetime()

