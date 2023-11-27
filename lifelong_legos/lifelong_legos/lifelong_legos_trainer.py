import torch
import rclpy
import numpy as np
from rclpy.node import Node
from lifelong_msgs.msg import ImagePose

from sensor_msgs.msg import Image
from geometry_msgs.msg import PoseStamped,Pose

def ros_pose_to_nerfstudio(pose_msg: Pose, static_transform=None):
    """
    Takes a ROS Pose message and converts it to the
    3x4 transform format used by nerfstudio.
    """
    quat = np.array(
        [
            pose_msg.orientation.w,
            pose_msg.orientation.x,
            pose_msg.orientation.y,
            pose_msg.orientation.z,
        ],
    )
    posi = torch.tensor([pose_msg.position.x, pose_msg.position.y, pose_msg.position.z])
    R = torch.tensor(qvec2rotmat(quat))
    T = torch.cat([R, posi.unsqueeze(-1)], dim=-1)
    T = T.to(dtype=torch.float32)
    if static_transform is not None:
        T = pose_utils.multiply(T, static_transform)
        T2 = torch.zeros(3, 4)
        R1 = transform.Rotation.from_euler("x", 90, degrees=True).as_matrix()
        R2 = transform.Rotation.from_euler("z", 180, degrees=True).as_matrix()
        R3 = transform.Rotation.from_euler("y", 180, degrees=True).as_matrix()
        R = torch.from_numpy(R3 @ R2 @ R1)
        T2[:, :3] = R
        T = pose_utils.multiply(T2, T)


    return T.to(dtype=torch.float32)

class TrainerNode(Node):
    def __init__(self,trainer):
        super().__init__('trainer_node')
        self.trainer_ = trainer
        self.subscription_ = self.create_subscription(ImagePose,"/camera/color/imagepose",self.add_img_callback,100)

    def add_img_callback(self,msg):
        print("Appending imagepose to queue",flush=True)
        self.trainer_.image_add_callback_queue.append(msg)
