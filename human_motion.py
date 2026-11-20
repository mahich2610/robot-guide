import threading
import time

import rclpy
from rclpy.executors import ExternalShutdownException, SingleThreadedExecutor
from rclpy.parameter import Parameter
from std_msgs.msg import Bool

from gz.msgs10.boolean_pb2 import Boolean
from gz.msgs10.pose_pb2 import Pose
from gz.transport13 import Node as GzNode

MODEL = 'human_static'
SERVICE = '/world/actors/set_pose'

START_X = -0.1

# скорость в м/с, длительность в секундах симуляции
STAGES = ((0.1, 10), (0.2, 10), (0.0, 5))

def move_human(gz_node, x, y):
    req = Pose()
    req.name = MODEL
    req.position.x = x
    req.position.y = y
    req.position.z = 0.0
    req.orientation.z = 0.707107
    req.orientation.w = 0.707107
    for _ in range(3):
        ok, reply = gz_node.request(SERVICE, req, Pose, Boolean, 1000)
        if ok and reply.data:
            return True
        time.sleep(0.5)

    return False


def main(args=None):
    gz_node = GzNode()
    rclpy.init(args=args)
    node = rclpy.create_node(
        'human_movement',
        parameter_overrides=[
            Parameter('use_sim_time', Parameter.Type.BOOL, True)])

    lateral_offset = node.declare_parameter('lateral_offset', 0.0).value
    ready = threading.Event()
    node.context.on_shutdown(ready.set)

    def on_ready(msg):
        (ready.set if msg.data else ready.clear)()

    node.create_subscription(Bool, '/speed_controller/ready', on_ready, 1)

    # отдельный поток принимает /clock, пока сценарий ждет ответа gazebo
    executor = SingleThreadedExecutor()
    executor.add_node(node)
    def spin():
        try:
            executor.spin()
        except ExternalShutdownException:
            pass
        except RuntimeError:
            if rclpy.ok():
                raise

    spin_thread = threading.Thread(target=spin)
    spin_thread.start()

    def sim_now():
        return node.get_clock().now().nanoseconds * 1e-9

    x = START_X

    try:
        if not ready.wait(timeout=30.0):
            return

        if not rclpy.ok():
            return

        if not move_human(gz_node, x, lateral_offset):
            return

        for speed, duration in STAGES:

            start_x, start_t = x, sim_now()

            while rclpy.ok():
                elapsed = min(sim_now() - start_t, duration)
                x = start_x + speed * elapsed

                if not move_human(gz_node, x, lateral_offset):
                    return

                if elapsed >= duration:
                    break

                time.sleep(0.02)

            if not rclpy.ok():
                return

    except (KeyboardInterrupt, ExternalShutdownException):
        pass
    except RuntimeError:
        if rclpy.ok():
            raise
    finally:
        executor.shutdown()
        spin_thread.join()
        node.destroy_node()
        rclpy.try_shutdown()


if __name__ == '__main__':
    main()