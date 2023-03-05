#!/bin/python3

import cv2 as cv
import numpy as np
import rclpy
from rclpy.node import Node
import time
import ament_index_python.packages
import os
from cv_bridge import CvBridge

from std_msgs.msg import Int8
from sensor_msgs.msg import Image
from yolo_msgs.msg import BoundingBoxes, BoundingBox

cfg_files_path = os.path.join(ament_index_python.packages.get_package_share_directory('foxy_yolo'), 'net_props/')

class YoloPublisher(Node):

    def __init__(self):
        super().__init__('yolo_node')
        self.detection_image_pub_ = self.create_publisher(Image, 'obj_rec_detection_image', 10)
        self.counted_objects_pub_ = self.create_publisher(Int8, "obj_rec_objects_counted", 10)
        self.bounding_boxes_pub_ = self.create_publisher(BoundingBoxes, "obj_rec_bounding_boxes", 10)

        self.camera_read_sub_ = self.create_subscription(Image, "/head_front_camera/rgb/image_raw", self.camera_read_callback, 10)

        self.net = cv.dnn.readNet(cfg_files_path + "yolo.weights", cfg_files_path + "yolo.cfg", "Darknet")
        self.classes = []
        with open(cfg_files_path + "classes.names", "r") as f:
            self.classes = [line.strip() for line in f.readlines()]
        
        self.layer_names = self.net.getLayerNames()
        self.output_layers = [self.layer_names[i - 1] for i in self.net.getUnconnectedOutLayers()]
        self.colors = np.random.uniform(0, 255, size=(len(self.classes), 3))
        self.frame_id = 0
        self.font = cv.FONT_HERSHEY_PLAIN
        self.starting_time = time.time()

        self.capture = cv.VideoCapture(0)

    
    def camera_read_callback(self, image):
        counted_objects = Int8()
        output_image = Image()
        bounding_boxes = BoundingBoxes()

        bounding_boxes.header.frame_id = str(self.frame_id)
        bounding_boxes.header.stamp = self.get_clock().now().to_msg()
        # bounding_boxes.image_header = image.header
        bounding_boxes.bounding_boxes = []


        ret, frame = self.capture.read()

        cv_image = frame

        bridge = CvBridge()

        height, width, channels = cv_image.shape
        self.frame_id += 1

        blob = cv.dnn.blobFromImage(cv_image, 0.00392, (416, 416), (0, 0, 0), True, crop=False)
        self.net.setInput(blob)
        outs = self.net.forward(self.output_layers)

        if self.frame_id == 3:
            print('outs', outs)

        class_ids = []
        confidences = []
        boxes = []
        for out in outs:
            for detection in out:
                scores = detection[5:]
                class_id = np.argmax(scores)
                confidence = scores[class_id]
                if confidence > 0.2:
                    # Object detected
                    center_x = int(detection[0] * width)
                    center_y = int(detection[1] * height)
                    w = int(detection[2] * 416)
                    h = int(detection[3] * 416)

                    # Rectangle coordinates
                    x = int(center_x - w / 2)
                    y = int(center_y - h / 2)

                    boxes.append([x, y, w, h])
                    confidences.append(float(confidence))
                    class_ids.append(class_id)
                    counted_objects.data += 1
                    bounding_box = BoundingBox()

                    bounding_box.class_name = self.classes[class_id]
                    bounding_box.probability = float(confidence)
                    bounding_box.xmin = x
                    bounding_box.ymin = y
                    bounding_box.xmax = x + w
                    bounding_box.ymax = y + h

                    bounding_boxes.bounding_boxes.append(bounding_box)
        
        indexes = cv.dnn.NMSBoxes(boxes, confidences, 0.5, 0.4)
        for i in range(len(boxes)):
            if i in indexes:
                x, y, w, h = boxes[i]
                label = str(self.classes[class_ids[i]])
                confidence = confidences[i]
                color = self.colors[i]
                cv.rectangle(cv_image, (x, y), (x + w, y + h), color, 2)
                cv.putText(cv_image, label + " " + str(round(confidence, 2)), (x, y + 30), self.font, 3, color, 3)

                print("Object detected: " + label + " " + str(round(confidence, 2)))

        cv.namedWindow("Image", cv.WINDOW_NORMAL)
        cv.imshow("Image", cv_image)
        cv.waitKey(1)
        
        output_image = bridge.cv2_to_imgmsg(cv_image, encoding="passthrough")

        self.counted_objects_pub_.publish(counted_objects)
        self.detection_image_pub_.publish(output_image)
        self.bounding_boxes_pub_.publish(bounding_boxes)


def main(args=None):
    rclpy.init(args=args)

    yolo_publisher = YoloPublisher()

    while True:
        YoloPublisher.camera_read_callback(yolo_publisher, Image())

    # Destroy the node explicitly
    # (optional - otherwise it will be done automatically
    # when the garbage collector destroys the node object)
    yolo_publisher.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    main()