# Specify model imports
from object_detection.builders import model_builder
from object_detection.utils import config_util
from object_detection.utils import label_map_util
from object_detection.utils import visualization_utils as viz_utils
from PIL import Image
import cv2
import numpy as np
import os
import tensorflow.compat.v1 as tf

path2config ='/Users/aryan/Desktop/RESEARCH FALL 2021/capuchin_obj_detection/exported_final_inference_model/my_model/pipeline.config'
path2model = '/Users/aryan/Desktop/RESEARCH FALL 2021/capuchin_obj_detection/exported_final_inference_model/my_model/checkpoint'
path2label_map = '/Users/aryan/Desktop/RESEARCH FALL 2021/capuchin_obj_detection/exported_final_inference_model/my_model/mscoco_label_map.pbtxt' # TODO: provide a path to the label map file
configs = config_util.get_configs_from_pipeline_file(path2config) # importing config
model_config = configs['model'] # recreating model config
detection_model = model_builder.build(model_config=model_config, is_training=False) # importing model

def detect_fn(image):
    """
    Detect objects in image.
    
    Args:
      image: (tf.tensor): 4D input image
      
    Returs:
      detections (dict): predictions that model made
    """

    image, shapes = detection_model.preprocess(image)
    prediction_dict = detection_model.predict(image, shapes)
    detections = detection_model.postprocess(prediction_dict, shapes)
    return detections
    
def detect_image(path, output_path):
  
  ckpt = tf.compat.v2.train.Checkpoint(model=detection_model)
  ckpt.restore(os.path.join(path2model, 'ckpt-0')).expect_partial()
  category_index = label_map_util.create_category_index_from_labelmap(path2label_map,use_display_name=True)
  image_np = cv2.imread(path)  
  image_np = cv2.cvtColor(image_np, cv2.COLOR_BGR2RGB)
  # print(image_np)
  # print(np.array(Image.open(path)))
  input_tensor = tf.convert_to_tensor(np.expand_dims(image_np, 0), dtype=tf.float32)
  detections = detect_fn(input_tensor)

  # All outputs are batches tensors.
  # Convert to numpy arrays, and take index [0] to remove the batch dimension.
  # We're only interested in the first num_detections.
  num_detections = int(detections.pop('num_detections'))
  detections = {key: value[0, :num_detections].numpy()
                for key, value in detections.items()}
  
  detections['num_detections'] = num_detections

  # detection_classes should be ints.
  detections['detection_classes'] = detections['detection_classes'].astype(np.int64)

  label_id_offset = 1
  image_np_with_detections = image_np.copy()

  viz_utils.visualize_boxes_and_labels_on_image_array(
          image_np_with_detections,
          detections['detection_boxes'],
          detections['detection_classes']+label_id_offset,
          detections['detection_scores'],
          category_index,
          use_normalized_coordinates=True,
          max_boxes_to_draw=200,
          min_score_thresh=0.4,
          agnostic_mode=False,
          line_thickness=2)
  cv2.imwrite(output_path, image_np_with_detections)

def detect_video(path, output_path):
  ckpt = tf.compat.v2.train.Checkpoint(model=detection_model)
  ckpt.restore(os.path.join(path2model, 'ckpt-0')).expect_partial()
  category_index = label_map_util.create_category_index_from_labelmap(path2label_map,use_display_name=True)
  # Set output video writer with codec
  fourcc = cv2.VideoWriter_fourcc(*'mp4v')
  out = cv2.VideoWriter(output_path, fourcc, 25.0, (1920, 1080))
  
  # Read the video
  vidcap = cv2.VideoCapture(path)
  frame_read, image_np = vidcap.read()
  # if not isBGR:
  #   image_np = cv2.cvtColor(image_np, cv2.COLOR_BGR2RGB)
  count = 0
  # Iterate over frames and pass each for prediction
  while frame_read:
    image_np = cv2.cvtColor(image_np, cv2.COLOR_BGR2RGB)
    # Perform object detection and add to output file
    input_tensor = tf.convert_to_tensor(np.expand_dims(image_np, 0), dtype=tf.float32)
    detections = detect_fn(input_tensor)

    # All outputs are batches tensors.
    # Convert to numpy arrays, and take index [0] to remove the batch dimension.
    # We're only interested in the first num_detections.
    num_detections = int(detections.pop('num_detections'))
    detections = {key: value[0, :num_detections].numpy()
                  for key, value in detections.items()}
    
    detections['num_detections'] = num_detections

    # detection_classes should be ints.
    detections['detection_classes'] = detections['detection_classes'].astype(np.int64)

    label_id_offset = 1
    image_np_with_detections = image_np.copy()

    viz_utils.visualize_boxes_and_labels_on_image_array(
            image_np_with_detections,
            detections['detection_boxes'],
            detections['detection_classes']+label_id_offset,
            detections['detection_scores'],
            category_index,
            use_normalized_coordinates=True,
            max_boxes_to_draw=200,
            min_score_thresh=0.4,
            agnostic_mode=False,
            line_thickness=2)
    print(str(count) + " frame inferred")
    # Write frame with predictions to video
    out.write(image_np_with_detections)
    
    # Read next frame
    frame_read, image_np = vidcap.read()
    
    count += 1
      
  # Release video file when we're ready
  out.release()

if __name__ == '__main__':
  # detector = TFObjectDetector('/Users/aryan/Desktop/RESEARCH FALL 2021/capuchin_obj_detection/exported_final_inference_model/my_model', '/Users/aryan/Desktop/RESEARCH FALL 2021/capuchin_obj_detection/exported_final_inference_model/my_model/checkpoint', '/Users/aryan/Desktop/RESEARCH FALL 2021/capuchin_obj_detection/exported_final_inference_model/my_model/mscoco_label_map.pbtxt', 'ssd_mobilenet_v2_fpnlite_640x640_coco17_tpu-8')
  detect_image('/Users/aryan/Desktop/jol.png', './out_46_test_lol.jpg')
  # detect_video('/Users/aryan/Desktop/RESEARCH FALL 2021/capuchin_obj_detection/S4c.7.18.19.MP4', './42k_inference_S4c.mp4')
