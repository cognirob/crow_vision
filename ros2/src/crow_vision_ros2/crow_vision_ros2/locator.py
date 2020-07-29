import rclpy
from rclpy.node import Node
from rcl_interfaces.msg import ParameterType
from ros2param.api import call_get_parameters
import message_filters
from crow_msgs.msg import DetectionMask
from sensor_msgs.msg import PointCloud2
from rclpy.qos import qos_profile_sensor_data
from rclpy.qos import QoSProfile
from rclpy.qos import QoSReliabilityPolicy


class Locator(Node):

    def __init__(self, node_name="locator_node"):
        super().__init__(node_name)
        self.image_topics, self.cameras = [p.string_array_value for p in call_get_parameters(node=self, node_name="/calibrator", parameter_names=["image_topics", "camera_nodes"]).values]
        self.mask_topics = [cam + "/" + "detections/masks" for cam in self.cameras]
        self.pcl_topics = [cam + "/" + "pointcloud" for cam in self.cameras]

        qos_profile = QoSProfile(
            depth=10,
            reliability=QoSReliabilityPolicy.RMW_QOS_POLICY_RELIABILITY_RELIABLE)
        for cam, pclTopic, maskTopic in zip(self.cameras, self.pcl_topics, self.mask_topics):
            self.subPCL = message_filters.Subscriber(self, PointCloud2, pclTopic, qos_profile=10)
            self.subMasks = message_filters.Subscriber(self, DetectionMask, maskTopic, qos_profile=10)
            self.sync = message_filters.ApproximateTimeSynchronizer([self.subPCL, self.subMasks], 20, 0.005)
            self.sync.registerCallback(lambda pcl_msg, mask_msg, cam=cam: self.detection_callback(pcl_msg, mask_msg, cam))
            print(self.sync.queue_size)

    def detection_callback(self, pcl_msg, mask_msg, camera):
        print("Got messages!!!!")


def main():
    rclpy.init()

    locator = Locator()

    rclpy.spin(locator)
    locator.destroy_node()


if __name__ == "__main__":
    main()
