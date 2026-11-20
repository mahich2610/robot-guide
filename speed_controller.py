import math
from collections import deque
from itertools import islice
from statistics import median

import rclpy
from rclpy.clock import JumpThreshold
from rclpy.duration import Duration
from rclpy.executors import ExternalShutdownException
from rclpy.node import Node
from std_msgs.msg import Bool, Float32
from geometry_msgs.msg import Twist, PointStamped, Vector3
from nav_msgs.msg import Odometry


class KalmanFilter1D:
    # в self.x лежат положение (м) и скорость (м/с)

    def __init__(self, q, r):
        self.q = q      # чем больше q, тем быстрее фильтр реагирует на смену темпа
        self.r = r      # чем больше r, тем меньше фильтр доверяет замеру положения
        self.reset()

    def reset(self):
        self.x = [0.0, 0.0]
        self.P = [[1.0, 0.0], [0.0, 1.0]]
        self.t = None
        self.initialized = False

    def predict(self, t):
        dt = t - self.t
        if dt <= 0.0:
            return

        p00 = self.P[0][0]
        p01 = self.P[0][1]
        p11 = self.P[1][1]
        q = self.q

        self.x[0] += self.x[1] * dt

        self.P[0][0] = p00 + 2.0 * dt * p01 + dt * dt * p11 + 0.25 * dt ** 4 * q
        self.P[0][1] = p01 + dt * p11 + 0.5 * dt ** 3 * q
        self.P[1][0] = self.P[0][1]
        self.P[1][1] = p11 + dt * dt * q

        self.t = t

    def update(self, t, z):
        # по одному замеру скорость еще не оценить
        if not self.initialized:
            self.x = [z, 0.0]
            self.P = [[self.r, 0.0], [0.0, 0.25]]
            self.t = t
            self.initialized = True
            return

        self.predict(t)

        innovation = z - self.x[0]
        s = self.P[0][0] + self.r

        k0 = self.P[0][0] / s
        k1 = self.P[1][0] / s

        self.x[0] += k0 * innovation
        self.x[1] += k1 * innovation

        p00 = self.P[0][0]
        p01 = self.P[0][1]
        p11 = self.P[1][1]

        self.P[0][0] = (1.0 - k0) * p00
        self.P[0][1] = (1.0 - k0) * p01
        self.P[1][0] = self.P[0][1]
        self.P[1][1] = p11 - k1 * p01


    def extrapolate(self, t):
        return self.x[0] + self.x[1] * max(0.0, t - self.t)


class SpeedController(Node):
    DATA_TIMEOUT = 0.7
    SAMPLE_TIMEOUT = 1.2
    MOTION_DATA_TIMEOUT = 0.35
    MOTION_WINDOW = 0.3
    STOP_SPEED = 0.025
    RESUME_SPEED = 0.045
    BRAKING_MARGIN = 0.035
    DISTANCE_HOLD_BAND = 0.04
    SPEED_HOLD_BAND = 0.02
    CONTROL_PERIOD = 0.05
    SPEED_RESPONSE = 0.22
    BRAKE_RESPONSE = 0.10
    STOP_COMMAND_EPS = 0.001
    MAX_HEADING_CHANGE = math.radians(12.0)
    MAX_ANGULAR_SPEED = 0.18

    def __init__(self):
        super().__init__('speed_controller')

        defaults = [
            ('target_distance', 2.0), ('k', 0.5), ('max_speed', 0.30),
            ('max_acceleration', 0.25), ('max_deceleration', 1.0),
            ('acceleration_jerk', 1.2), ('braking_jerk', 3.0),
            ('prediction_time', 0.25), ('deadzone', 0.02),
            ('kf_q', 0.3), ('kf_r', 0.0025), ('measurement_delay', 0.0),
            ('speed_lpf_tau', 0.0), ('use_stamped_distance', False),
            ('position_topic', ''), ('base_frame', 'base_link'),
        ]
        cfg = {param.name: param.value for param in self.declare_parameters('', defaults)}
        positive = {'max_deceleration', 'acceleration_jerk', 'braking_jerk', 'kf_r'}
        invalid_config = any(
            not math.isfinite(cfg[name]) or cfg[name] < 0.0
            or (name in positive and cfg[name] == 0.0)
            for name, default in defaults if isinstance(default, float)
        )
        if invalid_config or not 0.3 < cfg['target_distance'] < 10.0:
            self.destroy_node()
            rclpy.try_shutdown()
            raise SystemExit
        self.cfg = cfg
        self.kf = KalmanFilter1D(cfg['kf_q'], cfg['kf_r'])
        position_topic, use_stamped = cfg['position_topic'], cfg['use_stamped_distance']

        self.hist = deque(maxlen=3)
        self.motion_history = deque()
        self.recent_speed = 0.0
        self.human_moving = False

        # буфер одометрии: (время, x, v)
        self.odom_buf = deque(maxlen=300)
        self.track_heading = None
        self.last_odom_arrival = None
        self.last_meas_arrival = None

        self.human_speed = 0.0
        self.robot_speed = 0.0
        self.command_acceleration = 0.0
        self.last_control_time = self.now_s()

        if position_topic:
            self.create_subscription(PointStamped, position_topic, self.position_callback, 10)
        elif use_stamped:
            self.create_subscription(
                PointStamped, '/human/distance_stamped',
                self.stamped_callback, 10)
        else:
            self.create_subscription(
                Float32, '/human/distance',
                self.distance_callback, 10)

        self.create_subscription(
            Odometry, '/model/fun_robot/odometry',
            self.odometry_callback, 10)

        self.publisher = self.create_publisher(
            Twist, '/model/fun_robot/cmd_vel', 10)
        self.ready_pub = self.create_publisher(Bool, '/speed_controller/ready', 1)

        self.clock_jump = self.get_clock().create_jump_callback(
            JumpThreshold(min_forward=None, min_backward=Duration(nanoseconds=-1),
                          on_clock_change=True),
            post_callback=self.reset_clock_epoch)
        self.create_timer(self.CONTROL_PERIOD, self.control)


    def now_s(self):
        return self.get_clock().now().nanoseconds * 1e-9

    @staticmethod
    def stamp_s(stamp):
        return stamp.sec + stamp.nanosec * 1e-9

    def odometry_callback(self, msg):
        pose = msg.pose.pose
        twist = msg.twist.twist
        x, y = pose.position.x, pose.position.y
        v, angular_speed = twist.linear.x, twist.angular.z
        q = pose.orientation
        values = (x, y, v, angular_speed, q.x, q.y, q.z, q.w)
        if not all(math.isfinite(value) for value in values):
            return
        norm = math.hypot(q.x, q.y, q.z, q.w)
        if norm < 1e-6:
            return

        now = self.now_s()
        t = self.stamp_s(msg.header.stamp)

        # из за метки из будущего следующие сообщения будут считаться старыми
        if t <= 0.0:
            t = now
        if not -0.05 <= now - t <= self.DATA_TIMEOUT:
            return

        if self.odom_buf and t <= self.odom_buf[-1][0]:
            return

        qx, qy, qz, qw = (value / norm for value in (q.x, q.y, q.z, q.w))
        heading = math.atan2(2.0 * (qw * qz + qx * qy), 1.0 - 2.0 * (qy*qy + qz*qz))
        if self.track_heading is None:
            self.track_heading = heading

        difference = heading - self.track_heading
        forward_cos = math.cos(difference)
        if (forward_cos < math.cos(self.MAX_HEADING_CHANGE)
                or abs(angular_speed) > self.MAX_ANGULAR_SPEED):
            self.odom_buf.clear()
            self.last_odom_arrival = None
            self.stop_and_reset()
            self.publisher.publish(Twist())
            return

        along_track = x * math.cos(self.track_heading) + y * math.sin(self.track_heading)
        self.odom_buf.append((t, along_track, v * forward_cos))
        self.last_odom_arrival = now

    def odom_x_at(self, t):
        buf = self.odom_buf
        if not buf:
            return None

        if t >= buf[-1][0]:
            t0, x0, v0 = buf[-1]
            return x0 + v0 * min(t - t0, 0.2)

        if t < buf[0][0]:
            return None
        for (t0, x0, _), (t1, x1, _) in zip(buf, islice(buf, 1, None)):
            if t <= t1:
                return x0 + (x1 - x0) * (t - t0) / (t1 - t0)
        return buf[0][1]

    def odom_now(self, now):
        t0, x0, v0 = self.odom_buf[-1]
        return x0 + v0 * min(max(0.0, now - t0), 0.2), v0

    def distance_callback(self, msg):
        now = self.now_s()
        self.process_distance(
            float(msg.data), now - self.cfg['measurement_delay'], now)

    def stamped_callback(self, msg):
        now = self.now_s()
        t = self.stamp_s(msg.header.stamp)
        if t <= 0.0:
            t = now - self.cfg['measurement_delay']
        self.process_distance(float(msg.point.x), t, now)

    def position_callback(self, msg):
        if msg.header.frame_id != self.cfg['base_frame']:
            return
        stamp = self.stamp_s(msg.header.stamp)
        if stamp > 0.0:
            self.process_distance(-float(msg.point.x), stamp, self.now_s())

    def process_distance(self, d, t_meas, now):
        if not math.isfinite(d) or not 0.3 < d < 10.0:
            return

        if not now - self.SAMPLE_TIMEOUT <= t_meas <= now + 0.05:
            return
        if self.hist and t_meas <= self.hist[-1][0]:
            return
        self.hist.append((t_meas, d))

        # дистанция берется по медиане трех замеров, время — по второму замеру
        if len(self.hist) == 3:
            d_f = median(v for _, v in self.hist)
            t_f = self.hist[1][0]
        else:
            d_f, t_f = d, t_meas

        if self.last_odom_arrival is None or now - self.last_odom_arrival > 0.5:
            return

        if self.kf.initialized and t_f <= self.kf.t:
            return

        x_r = self.odom_x_at(t_f)
        x_r_scan = x_r if t_f == t_meas else self.odom_x_at(t_meas)
        if x_r is None or x_r_scan is None:
            return

        # замер без сглаживания помог раньше заметить, что человек сбавил шаг
        self.update_motion(t_meas, x_r_scan - d)

        self.kf.update(t_f, x_r - d_f)
        self.last_meas_arrival = now

    def update_motion(self, t, x):
        history = self.motion_history
        history.append((t, x))
        while len(history) > 1 and t - history[1][0] >= self.MOTION_WINDOW:
            history.popleft()

        t0, x0 = history[0]
        elapsed = t - t0
        if elapsed < self.MOTION_WINDOW:
            return

        self.recent_speed = (x - x0) / elapsed
        threshold = self.STOP_SPEED if self.human_moving else self.RESUME_SPEED
        self.human_moving = self.recent_speed >= threshold

    def soft_deadzone(self, e):
        return math.copysign(max(0.0, abs(e) - self.cfg['deadzone']), e)

    def stop_and_reset(self):
        self.robot_speed = 0.0
        self.command_acceleration = 0.0
        self.human_speed = 0.0
        self.hist.clear()
        self.reset_motion()
        self.last_meas_arrival = None
        self.kf.reset()

    def reset_motion(self):
        self.motion_history.clear()
        self.recent_speed = 0.0
        self.human_moving = False

    def update_speed_command(self, change, dt):
        braking = change < 0.0
        stopping = braking and change <= -self.robot_speed
        response = self.BRAKE_RESPONSE if braking else self.SPEED_RESPONSE
        jerk = self.cfg['braking_jerk'] if braking else self.cfg['acceleration_jerk']
        desired_acceleration = max(-self.cfg['max_deceleration'], min(
            self.cfg['max_acceleration'], change / response))
        acceleration_step = jerk * dt
        acceleration = self.command_acceleration + max(
            -acceleration_step, min(acceleration_step,
                                    desired_acceleration - self.command_acceleration))

        step = acceleration * dt
        # при торможении скорость не растет, а шаг не проходит мимо нужной скорости
        step = max(change, min(0.0, step)) if braking else max(0.0, min(change, step))

        self.command_acceleration = step / dt
        self.robot_speed += step
        if stopping and self.robot_speed <= self.STOP_COMMAND_EPS:
            self.robot_speed = 0.0
            self.command_acceleration = 0.0

    def reset_clock_epoch(self, _jump):
        self.stop_and_reset()
        self.odom_buf.clear()
        self.track_heading = None
        self.last_odom_arrival = None
        self.last_control_time = self.now_s()
        self.ready_pub.publish(Bool(data=False))
        self.publisher.publish(Twist())

    def control(self):
        now = self.now_s()
        dt = now - self.last_control_time

        if dt <= 0.0:       # симуляция на паузе
            return

        self.last_control_time = now
        dt = min(dt, 2 * self.CONTROL_PERIOD)

        data_ok = (
            self.kf.initialized
            and self.last_meas_arrival is not None
            and now - self.last_meas_arrival <= self.DATA_TIMEOUT
            and now - self.kf.t <= self.SAMPLE_TIMEOUT
            and self.last_odom_arrival is not None
            and now - self.last_odom_arrival <= self.DATA_TIMEOUT
            and now - self.odom_buf[-1][0] <= self.DATA_TIMEOUT
        )

        self.ready_pub.publish(Bool(data=data_ok))
        if not data_ok:
            self.stop_and_reset()
            self.publisher.publish(Twist())
            return

        x_h = self.kf.extrapolate(now)
        v_raw = self.kf.x[1]
        v_est = max(0.0, min(self.cfg['max_speed'], v_raw))

        tau = self.cfg['speed_lpf_tau']
        if tau > 0.0:
            a = dt / (tau + dt)
            self.human_speed += a * (v_est - self.human_speed)
        else:
            self.human_speed = v_est

        x_r, v_r = self.odom_now(now)

        d_now = x_r - x_h
        d_pred = d_now + (v_r - self.human_speed) * self.cfg['prediction_time']

        error = self.soft_deadzone(self.cfg['target_distance'] - d_pred)

        target = self.human_speed + self.cfg['k'] * error
        # после паузы оценка скорости в фильтре может устареть
        if (self.motion_history
                and now - self.motion_history[-1][0] > self.MOTION_DATA_TIMEOUT):
            self.reset_motion()

        if not self.human_moving:
            target = 0.0
        elif self.recent_speed + self.BRAKING_MARGIN < self.human_speed:
            # свежие замеры показывают, что человек сбавил шаг - робот прекращает разгон
            target = min(target, self.robot_speed, self.recent_speed)

        target = max(0.0, min(self.cfg['max_speed'], target))

        change = target - self.robot_speed
        # вблизи целевой дистанции разгон возможен только при свежей оценке более высокого темпа
        if (change > 0.0
                and abs(d_pred - self.cfg['target_distance']) <= self.DISTANCE_HOLD_BAND
                and self.recent_speed <= self.robot_speed + self.SPEED_HOLD_BAND):
            change = 0.0

        self.update_speed_command(change, dt)

        self.publisher.publish(Twist(linear=Vector3(x=float(self.robot_speed))))


def main(args=None):
    rclpy.init(args=args)
    node = SpeedController()

    try:
        rclpy.spin(node)
    except (KeyboardInterrupt, ExternalShutdownException):
        pass
    except RuntimeError:
        if rclpy.ok():
            raise
    finally:
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == '__main__':
    main()