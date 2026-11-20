from concurrent.futures import ThreadPoolExecutor

import cv2
import numpy as np
from numpy.lib.recfunctions import structured_to_unstructured
import rclpy
import torch

from rclpy.clock import JumpThreshold
from rclpy.duration import Duration
from rclpy.executors import ExternalShutdownException
from rclpy.node import Node
from geometry_msgs.msg import Point, PointStamped
from rclpy.time import Time
from rclpy.qos import QoSProfile, ReliabilityPolicy
from sensor_msgs.msg import CameraInfo, Image, PointCloud2
from std_msgs.msg import Float32, Header
from sensor_msgs_py import point_cloud2
from tf2_ros import Buffer, TransformException, TransformListener
from tf2_sensor_msgs.tf2_sensor_msgs import transform_points
from ultralytics import YOLO


class HumanTracking(Node):
    MAX_DETECTION_AGE = 0.35
    BOX_MARGIN = 12
    ZERO_VECTOR = np.zeros(3)
    DISTORTION_LENGTHS = {
        '': (0,), 'plumb_bob': (0, 5),
        'rational_polynomial': (8,), 'equidistant': (4,),
    }

    def __init__(self):
        super().__init__('human_tracking')

        # лимит потоков yolo оставляет ресурсы для датчиков gazebo
        torch.set_num_threads(2)
        torch.set_num_interop_threads(1)
        cv2.setNumThreads(1)
        self.model = YOLO('yolo11n.pt')
        self.inference_size = self.declare_parameter('inference_size', 640).value

        self.show_window = self.declare_parameter('show_window', True).value

        self.detection = None
        self.last_scan_stamp = None
        self.last_distance = None
        self.camera_info = None
        self.inference_worker = ThreadPoolExecutor(
            max_workers=1, initializer=lambda: torch.set_num_threads(2))
        self.pending_image = None
        self.inference_job = None
        self.frame_epoch = 0
        self.tf_buffer = Buffer()
        self.tf_listener = TransformListener(self.tf_buffer, self)

        image_topic = self.declare_parameter('image_topic', '/camera/image_raw').value
        info_topic = self.declare_parameter('camera_info_topic', '/camera/camera_info').value
        cloud_topic = self.declare_parameter('cloud_topic', '/lidar/scan/points').value
        self.min_lidar_z = self.declare_parameter('min_lidar_z', -0.10).value
        self.base_frame = self.declare_parameter('base_frame', 'base_link').value
        latest_sensor = QoSProfile(depth=1, reliability=ReliabilityPolicy.BEST_EFFORT)
        self.create_subscription(Image, image_topic, self.image_callback, latest_sensor)
        self.create_subscription(CameraInfo, info_topic, self.info_callback, latest_sensor)
        self.create_subscription(PointCloud2, cloud_topic, self.lidar_callback, latest_sensor)

        self.distance_pub = self.create_publisher(Float32, '/human/distance', 10)
        self.stamped_pub = self.create_publisher(PointStamped, '/human/distance_stamped', 10)
        self.position_pub = self.create_publisher(PointStamped, '/human/position', 10)
        self.clock_jump = self.get_clock().create_jump_callback(
            JumpThreshold(min_forward=None, min_backward=Duration(nanoseconds=-1),
                          on_clock_change=True),
            post_callback=self.reset_clock_epoch)
        self.create_timer(0.02, self.poll_inference)


    def clock_s(self):
        return self.get_clock().now().nanoseconds * 1e-9

    @staticmethod
    def stamp_s(stamp):
        return stamp.sec + stamp.nanosec * 1e-9

    def invalidate_detection(self):
        self.frame_epoch += 1
        self.pending_image = self.detection = None

    def reset_clock_epoch(self, _jump):
        self.invalidate_detection()
        self.last_scan_stamp = None
        self.last_distance = None

    def info_callback(self, msg):
        self.camera_info = msg

    def project_points(self, points, info):
        intrinsics = np.asarray(info.k, dtype=np.float64).reshape(3, 3)
        distortion = np.asarray(info.d, dtype=np.float64)
        if info.distortion_model == 'equidistant':
            pixels, _ = cv2.fisheye.projectPoints(
                points.reshape(-1, 1, 3), self.ZERO_VECTOR, self.ZERO_VECTOR,
                intrinsics, distortion)
        else:
            pixels, _ = cv2.projectPoints(
                points, self.ZERO_VECTOR, self.ZERO_VECTOR, intrinsics, distortion)
        return pixels.reshape(-1, 2)

    def lidar_callback(self, msg):
        stamp = msg.header.stamp
        scan_time = self.stamp_s(stamp)
        if not -0.05 <= self.clock_s() - scan_time <= self.MAX_DETECTION_AGE:
            return
        if self.last_scan_stamp is not None and scan_time <= self.stamp_s(self.last_scan_stamp):
            return
        self.last_scan_stamp = stamp
        self.last_distance = None

        if self.detection is None:
            return
        frame_time, (x1, y1, x2, y2), info = self.detection
        if abs(scan_time - frame_time) > self.MAX_DETECTION_AGE:
            return

        try:
            scan_stamp = Time.from_msg(stamp)
            to_camera = self.tf_buffer.lookup_transform(
                info.header.frame_id, msg.header.frame_id, scan_stamp)
            to_base = self.tf_buffer.lookup_transform(
                self.base_frame, msg.header.frame_id, scan_stamp)
        except TransformException:
            return

        xyz = point_cloud2.read_points(msg, field_names=('x', 'y', 'z'), skip_nans=True)
        points = structured_to_unstructured(xyz).reshape(-1, 3)
        # точки пола могут попасть в рамку человека
        points = points[np.isfinite(points).all(axis=1) & (points[:, 2] >= self.min_lidar_z)]
        optical_points = transform_points(points, to_camera.transform)
        in_front = optical_points[:, 2] > 0.35
        if not np.any(in_front):
            return

        pixels = self.project_points(optical_points[in_front], info)
        u, v = pixels.T
        inside = (
            (u >= x1 - self.BOX_MARGIN) & (u <= x2 + self.BOX_MARGIN)
            & (v >= y1 - self.BOX_MARGIN) & (v <= y2 + self.BOX_MARGIN)
        )
        candidates = points[in_front][inside]
        if len(candidates) < 3:
            return

        ranges = np.linalg.norm(candidates, axis=1)
        order = np.argsort(ranges)
        boundaries = np.flatnonzero(np.diff(ranges[order]) > 0.25) + 1
        group = max(np.split(order, boundaries), key=len)
        if len(group) < 3:
            return

        x, y, z = np.median(transform_points(candidates[group], to_base.transform), axis=0)
        if x >= 0.0:
            return
        self.position_pub.publish(PointStamped(
            header=Header(stamp=stamp, frame_id=self.base_frame),
            point=Point(x=float(x), y=float(y), z=float(z)),
        ))

        distance = float(np.median(ranges[group]))
        self.last_distance = distance
        self.distance_pub.publish(Float32(data=distance))
        self.stamped_pub.publish(PointStamped(
            header=msg.header, point=Point(x=distance),
        ))

    def image_callback(self, msg):
        info = self.camera_info
        if (info is None or not info.header.frame_id or info.k[0] <= 0.0
                or info.k[4] <= 0.0 or info.header.frame_id != msg.header.frame_id
                or (info.width, info.height) != (msg.width, msg.height)
                or len(info.d) not in self.DISTORTION_LENGTHS.get(info.distortion_model, ())):
            self.invalidate_detection()
            return
        if (msg.encoding not in ('rgb8', 'bgr8') or msg.step < 3 * msg.width
                or len(msg.data) != msg.height * msg.step):
            self.invalidate_detection()
            return

        # в очереди один свежий кадр, иначе растет задержка
        self.pending_image = (msg, info, self.frame_epoch)

    def poll_inference(self):
        if self.inference_job is not None:
            if not self.inference_job.done():
                return
            job, self.inference_job = self.inference_job, None
            try:
                epoch, data = job.result()
            except Exception:
                self.detection = None
            else:
                if epoch == self.frame_epoch:
                    self.handle_detection(*data)

        if self.pending_image is not None:
            self.inference_job = self.inference_worker.submit(self.run_inference, *self.pending_image)
            self.pending_image = None

    def run_inference(self, msg, info, epoch):
        data = np.frombuffer(msg.data, dtype=np.uint8).reshape(msg.height, msg.step)
        image = np.ascontiguousarray(data[:, :msg.width * 3]).reshape(msg.height, msg.width, 3)

        if msg.encoding == 'rgb8':
            image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

        result = self.model.predict(
            image, imgsz=self.inference_size, classes=[0],
            conf=0.25, verbose=False, device='cpu',
        )[0]

        # первый запуск yolo может сбить лимит, поэтому снова ставится 2 потока
        if torch.get_num_threads() != 2:
            torch.set_num_threads(2)
        return epoch, (msg, info, result)

    def handle_detection(self, msg, info, result):
        # сопровождение только при одной рамке, иначе можно захватить прохожего
        self.detection = None
        if len(result.boxes) == 1:
            box = result.boxes[0]
            frame_time = self.stamp_s(msg.header.stamp)
            self.detection = (frame_time, box.xyxy[0].tolist(), info)

        if self.show_window:
            output = result.plot()
            if self.detection is not None:
                fresh = (self.last_scan_stamp is not None
                         and abs(frame_time - self.stamp_s(self.last_scan_stamp))
                         <= self.MAX_DETECTION_AGE)
                label = (f'{round(self.last_distance, 2)} м'
                         if fresh and self.last_distance is not None else 'нет данных')
                cv2.putText(output, f'Distance: {label}', (10, 25),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)
            cv2.imshow('YOLO + 3D LiDAR', output)
            cv2.waitKey(1)


def main(args=None):
    rclpy.init(args=args)
    node = HumanTracking()

    try:
        rclpy.spin(node)

    except (KeyboardInterrupt, ExternalShutdownException):
        pass
    except RuntimeError:
        if rclpy.ok():
            raise

    finally:
        node.inference_worker.shutdown(cancel_futures=True)
        node.destroy_node()

        if rclpy.ok():
            rclpy.shutdown()

        cv2.destroyAllWindows()


if __name__ == '__main__':
    main()