#!/bin/python3

import cv2 as cv
import cv_bridge
import numpy as np
import rclpy
from rclpy.node import Node
from message_filters import ApproximateTimeSynchronizer, Subscriber
import rclpy.logging
import sys


from sensor_msgs.msg import Image
from yolo_msgs.msg import BoundingBoxes, BoundingBox


class TrackingNode(Node):

    def __init__(self):
        super().__init__('tracking_node')
        
        self.image_sub_ = Subscriber(self, Image, "/head_front_camera/rgb/image_raw")
        self.bounding_boxes_sub_ = Subscriber(self, BoundingBoxes, "/obj_rec/bounding_boxes")

        self.init_tracking_sub_synchronizer = ApproximateTimeSynchronizer([self.image_sub_, self.bounding_boxes_sub_], 10, 1)
        self.init_tracking_sub_synchronizer.registerCallback(self.init_track_callback)

        self.tracker = cv.TrackerKCF_create()

        self.cv_bridge = cv_bridge.CvBridge()

        self.init_tracker = False

    
    def init_track_callback(self, image: Image, bounding_boxes: BoundingBoxes):
        for bounding_box in bounding_boxes.bounding_boxes:
            if bounding_box.id == 0:
                rclpy.logging.get_logger('TrackingPublisher').info('Found a person!')

                cv_image = self.cv_bridge.imgmsg_to_cv2(image)
                cv_image = cv.resize(cv_image, (640, 480))
                height, width, _ = cv_image.shape

                x = int(bounding_box.xmin * width)
                y = int(bounding_box.ymin * height)
                w = int(bounding_box.xmax * width) - x
                h = int(bounding_box.ymax * height) - y

                self.init_tracker = self.tracker.init(cv_image, (x, y, w, h))

                self.init_tracking_sub_synchronizer.unregisterCallback(self.init_track_callback)
                self.image_sub_.register_callback(self.track_callback)

                
                    
    def track_callback(self, image: Image):
        if self.init_tracker is not None:
            cv_image = self.cv_bridge.imgmsg_to_cv2(image)
            cv_image = cv.resize(cv_image, (640, 480))

            success, box = self.tracker.update(cv_image)

            if success:
                x, y, w, h = [int(v) for v in box]
                cv.rectangle(cv_image, (x, y), (x + w, y + h), (255, 0, 0), 2)

            cv.imshow('Tracking', cv_image)
            cv.waitKey(1)


def main(args=None):
    print("OpenCV version: ", cv.__version__)
    print("Python environment: ", sys.executable)

    rclpy.init(args=args)

    tracking_node = TrackingNode()

    rclpy.spin(tracking_node)

    # Destroy the node explicitly
    # (optional - otherwise it will be done automatically
    # when the garbage collector destroys the node object)
    tracking_node.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    main()