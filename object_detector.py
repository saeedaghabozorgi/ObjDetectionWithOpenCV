import numpy as np
import os
import six.moves.urllib as urllib
import sys
import tarfile
import tensorflow as tf
import zipfile

from collections import defaultdict
from io import StringIO
from matplotlib import pyplot as plt
from PIL import Image

from object_detection.utils import label_map_util

from object_detection.utils import visualization_utils as vis_util
from cv2 import *
def download_model(MODEL_FILE):
    #opener = urllib.request.URLopener()
    #opener.retrieve(DOWNLOAD_BASE + MODEL_FILE, MODEL_FILE)
    MODEL_FILE = '/tmp/ModelZoo/ssd_inception_v2_coco_11_06_2017.tar.gz'
    tar_file = tarfile.open(MODEL_FILE)
    for file in tar_file.getmembers():
        file_name = os.path.basename(file.name)
        if 'frozen_inference_graph.pb' in file_name:
            tar_file.extract(file, os.getcwd())

def load_labelmap(PATH_TO_LABELS, NUM_CLASSES = 90): 
    # Label maps map indices to category names, so that when our convolution network predicts 5, we know that this corresponds to airplane. Here we use internal utility functions, but anything that returns a dictionary mapping integers to appropriate string labels would be fine
    label_map = label_map_util.load_labelmap(PATH_TO_LABELS)
    categories = label_map_util.convert_label_map_to_categories(label_map, max_num_classes=NUM_CLASSES, use_display_name=True)
    category_index = label_map_util.create_category_index(categories)
    return category_index

def Model_loader(PATH_TO_CKPT):
    # Load a (frozen) Tensorflow model into memory.
    detection_graph = tf.Graph()
    with detection_graph.as_default():
        od_graph_def = tf.GraphDef()
        with tf.gfile.GFile(PATH_TO_CKPT, 'rb') as fid:
            serialized_graph = fid.read()
            od_graph_def.ParseFromString(serialized_graph)
            tf.import_graph_def(od_graph_def, name='')
    return detection_graph

def load_image_into_numpy_array(image):
  (im_width, im_height) = image.size
  return np.array(image.getdata()).reshape((im_height, im_width, 3)).astype(np.uint8)

def detect_objects(image_np, sess, detection_graph):
    

    # Expand dimensions since the model expects images to have shape: [1, None, None, 3]
    image_np_expanded = np.expand_dims(image_np, axis=0)
    image_tensor = detection_graph.get_tensor_by_name('image_tensor:0')

    # Each box represents a part of the image where a particular object was detected.
    boxes = detection_graph.get_tensor_by_name('detection_boxes:0')

    # Each score represent how level of confidence for each of the objects.
    # Score is shown on the result image, together with the class label.
    scores = detection_graph.get_tensor_by_name('detection_scores:0')
    classes = detection_graph.get_tensor_by_name('detection_classes:0')
    num_detections = detection_graph.get_tensor_by_name('num_detections:0')

    # Actual detection.
    (boxes, scores, classes, num_detections) = sess.run(
        [boxes, scores, classes, num_detections],
        feed_dict={image_tensor: image_np_expanded})
    
    # List of the strings that is used to add correct label for each box.
    PATH_TO_LABELS = os.path.join('object_detection', 'data', 'mscoco_label_map.pbtxt')
    category_index = load_labelmap(PATH_TO_LABELS, 90)

    # Visualization of the results of a detection.
    vis_util.visualize_boxes_and_labels_on_image_array(
        image_np,
        np.squeeze(boxes),
        np.squeeze(classes).astype(np.int32),
        np.squeeze(scores),
        category_index,
        use_normalized_coordinates=True,line_thickness=8)
    return image_np

def detect_one_file(sess,detection_graph):
    # read one file
    image_file = './object_detection/test_images/image1.jpg'
    image = Image.open(image_file)
    # the array based representation of the image will be used later in order to prepare the
    # result image with boxes and labels on it.
    image_np = load_image_into_numpy_array(image)
    image_np = detect_objects(image_np, sess, detection_graph)

    img = Image.fromarray(image_np, 'RGB')
    img.save('my.png')
    img.show()

def detect_one_file_userinput(sess,detection_graph):
    while True:
        input_var = input("Enter something: ")
        print ("you entered " + input_var) 
        image_file = './object_detection/test_images/image1.jpg'
        image = Image.open(image_file)
        # the array based representation of the image will be used later in order to prepare the
        # result image with boxes and labels on it.
        image_np = load_image_into_numpy_array(image)
        image_np = detect_objects(image_np, sess, detection_graph)

        img = Image.fromarray(image_np, 'RGB')
        img.save('my_image.png')
        img.show()

def detect_one_file_webcam(sess,detection_graph):
    # initialize the camera
    cap = VideoCapture(0)   # 0 -> index of camera

    while(True):
        # Capture frame-by-frame
        ret, frame = cap.read()
        cv2.imshow('frame',frame)
        if cv2.waitKey(1) & 0xFF == ord('d'):
            #imwrite("filename.jpg",frame)
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            image_np = detect_objects(frame_rgb, sess, detection_graph)
            img = Image.fromarray(image_np, 'RGB')
            img.show()
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == '__main__':
    PATH_TO_CKPT = '/tmp/ModelZoo/ssd_inception_v2_coco_11_06_2017/frozen_inference_graph.pb'
    detection_graph = Model_loader(PATH_TO_CKPT)
    sess = tf.Session(graph=detection_graph)
    # detect_one_file(sess,detection_graph)
    # detect_one_file_userinput(sess,detection_graph)
    detect_one_file_webcam(sess,detection_graph)
    sess.close()