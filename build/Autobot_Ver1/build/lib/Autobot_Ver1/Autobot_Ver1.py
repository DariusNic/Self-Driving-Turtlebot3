#!/usr/bin/env python3
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image, LaserScan
from geometry_msgs.msg import Twist
from cv_bridge import CvBridge
import cv2
import numpy as np
import time

class LaneFollower(Node):
    def __init__(self):
        super().__init__('lane_follower_node')

        # Subscriber camera (pentru stop sign + linie galbenă)
        self.image_sub = self.create_subscription(
            Image,
            '/camera/image_raw',
            self.image_callback,
            10
        )

        # Subscriber LiDAR (pentru obstacol)
        self.scan_sub = self.create_subscription(
            LaserScan,
            '/scan',
            self.lidar_callback,
            10
        )

        self.cmd_vel_pub = self.create_publisher(Twist, '/cmd_vel', 10)
        self.bridge = CvBridge()

        # ---------------------------
        # Parametri lane-follow
        # ---------------------------
        self.linear_speed = 0.4
        self.angular_gain = 0.015
        self.filtered_error = 0.0
        self.alpha = 0.7

        # LiDAR
        self.latest_scan_ranges = None
        self.num_lidar_rays = 0
        # Prag pentru considerare obstacol în față
        self.lidar_threshold = 1.6

        # FSM stări
        self.LANE_FOLLOW = 0
        self.CROSS_MIDDLE_LINE = 1
        self.PASS_OBSTACLE = 2
        self.CROSS_BACK = 3

        self.fsm_state = self.LANE_FOLLOW
        self.state_start_time = time.time()

        # Timp viraj stânga (depășire amplă)
        self.turn_left_time = 1.6

        # DETECTAREA OBSTACOLULUI “CONSISTENT” ÎN LANE_FOLLOW
        # ex. 5 cadre consecutive
        self.obstacle_in_front_count = 0
        self.obstacle_in_front_count_needed = 10

        # În PASS_OBSTACLE: confirmare obstacol în spate
        self.obstacle_behind_count_needed = 10
        self.obstacle_behind_count = 0

        self.get_logger().info("Lane Follower node started (LiDAR obstacole, + consecutive frames).")

    # ---------------------------------------------------------------------
    # LiDAR
    # ---------------------------------------------------------------------
    def lidar_callback(self, scan_msg):
        self.latest_scan_ranges = scan_msg.ranges
        self.num_lidar_rays = len(scan_msg.ranges)

    # ---------------------------------------------------------------------
    # image_callback
    # ---------------------------------------------------------------------
    def image_callback(self, msg):
        cv_image = self.bridge.imgmsg_to_cv2(msg, desired_encoding='bgr8')

        # 1) STOP SIGN (opțional)
        '''
        if self.detect_stop_sign(cv_image):
            self.get_logger().info("STOP sign detected! Robot stops.")
            self.stop_robot()
            self.show_debug(cv_image, None)
            return
          '''
        # 2) FSM
        if self.fsm_state == self.LANE_FOLLOW:
            # Verificare obstacol cu “buffer” de cadre
            if self.lidar_obstacle_in_front():
                self.obstacle_in_front_count += 1
            else:
                self.obstacle_in_front_count = 0

            # Dacă am atins numărul minim de cadre consecutive
            if self.obstacle_in_front_count >= self.obstacle_in_front_count_needed:
                self.get_logger().info("Obstacle in front (x frames) => CROSS_MIDDLE_LINE.")
                self.fsm_state = self.CROSS_MIDDLE_LINE
                self.state_start_time = time.time()
            else:
                # Lane-follow normal
                self.lane_follow(cv_image)

        elif self.fsm_state == self.CROSS_MIDDLE_LINE:
            # Viraj stânga un timp
            if (time.time() - self.state_start_time) < self.turn_left_time:
                twist = Twist()
                twist.linear.x = 0.2
                twist.angular.z = 0.3
                self.cmd_vel_pub.publish(twist)
            else:
                self.get_logger().info("Done turning left => PASS_OBSTACLE.")
                self.fsm_state = self.PASS_OBSTACLE
                self.obstacle_behind_count = 0

        elif self.fsm_state == self.PASS_OBSTACLE:
            # Mergi inainte până obstacolul e în spate-dreapta x cadre
            if self.lidar_obstacle_in_right_rear():
                self.obstacle_behind_count += 1
                self.get_logger().info(f"Obstacle behind-dreapta frames= {self.obstacle_behind_count}")
            else:
                self.obstacle_behind_count = 0

            if self.obstacle_behind_count >= self.obstacle_behind_count_needed:
                self.get_logger().info("Obstacle in spate => CROSS_BACK.")
                self.fsm_state = self.CROSS_BACK
                self.state_start_time = time.time()
            else:
                twist = Twist()
                twist.linear.x = 0.3
                twist.angular.z = 0.0
                self.cmd_vel_pub.publish(twist)

        elif self.fsm_state == self.CROSS_BACK:
            line_x = self.detect_yellow_line_center(cv_image)
            width = cv_image.shape[1]
            center_x = width // 2
            OFFSET = 50  # Ajustează tu

            if line_x >= 0 and line_x < (center_x - OFFSET):
                self.get_logger().info("Line on left => back to LANE_FOLLOW.")
                self.fsm_state = self.LANE_FOLLOW
            else:
                 twist = Twist()
                 twist.linear.x = 0.25
                 twist.angular.z = -0.35
                 self.cmd_vel_pub.publish(twist)


        # Debug
        blur = cv2.blur(cv_image, (3,3))
        edge = cv2.Canny(blur, 160, 180)
        self.show_debug(cv_image, edge)

    # ---------------------------------------------------------------------
    # LANE FOLLOW
    # ---------------------------------------------------------------------
    def lane_follow(self, cv_image):
        blur = cv2.blur(cv_image, (3,3))
        edge = cv2.Canny(blur, 160, 180)
        cx = self.freespace(edge, cv_image)
        if cx >= 0:
            width = cv_image.shape[1]
            center_x = width // 2
            error = cx - center_x
            self.filtered_error = (self.alpha * self.filtered_error) + ((1 - self.alpha) * error)
            angular_z = -self.filtered_error * self.angular_gain
            twist = Twist()
            twist.linear.x = self.linear_speed
            twist.angular.z = angular_z
            self.cmd_vel_pub.publish(twist)
        else:
            self.stop_robot()

    def freespace(self, canny_frame, img):
        height, width = canny_frame.shape
        DreaptaLim = width // 2
        StangaLim = width // 2

        for i in range(width // 2, width-1):
            if canny_frame[height - 10, i]:
                DreaptaLim = i
                break
        for i in range(width // 2):
            if canny_frame[height - 10, width // 2 - i]:
                StangaLim = width // 2 - i
                break

        if StangaLim == width // 2:
            StangaLim = 1
        if DreaptaLim == width // 2:
            DreaptaLim = width - 1

        contour = []
        contour.append((StangaLim, height - 10))
        for j in range(StangaLim, DreaptaLim + 1, 10):
            for i in range(height - 10, 9, -1):
                if canny_frame[i, j]:
                    contour.append((j, i))
                    break
                if i == 10:
                    contour.append((j, i))
        contour.append((DreaptaLim, height - 10))
        contours = [np.array(contour)]
        M = cv2.moments(contours[0])
        if M["m00"] != 0:
            cx = int(M["m10"] / M["m00"])
            cy = int(M["m01"] / M["m00"])
            cv2.arrowedLine(img, (width//2, height-10), (cx, cy), (60,90,255), 4)
            return cx
        else:
            return -1

    # ---------------------------------------------------------------------
    # LIDAR: obstacol in fata?
    # ---------------------------------------------------------------------
    def lidar_obstacle_in_front(self):
        if self.latest_scan_ranges is None:
            return False
        n = self.num_lidar_rays
        if n == 0:
            return False

        # ±15° = 0..15 + n-15..n-1
        end_front = min(15, n)
        for i in range(end_front):
            if self.latest_scan_ranges[i] < self.lidar_threshold:
                return True
        start_back = max(n-15, 0)
        for i in range(start_back, n):
            if self.latest_scan_ranges[i] < self.lidar_threshold:
                return True

        return False

    # ---------------------------------------------------------------------
    # LIDAR: obstacol in spate-dreapta
    # ---------------------------------------------------------------------
    def lidar_obstacle_in_right_rear(self):
        """
        Return True daca TOT sectorul [300..359] + [0..30]
        e > self.lidar_threshold => nimic aproape => obstacol e behind-right
        """
        if self.latest_scan_ranges is None:
            return False
        n = len(self.latest_scan_ranges)
        if n == 0:
            return False

        ranges = self.latest_scan_ranges
        good = True

        # 300..359
        startA = 300
        if startA >= n:
            startA = n-1
        for i in range(startA, n):
            if ranges[i] < self.lidar_threshold:
                good = False
                break

        # 0..30
        if good:
            endB = min(30, n)
            for i in range(endB):
                if ranges[i] < self.lidar_threshold:
                    good = False
                    break

        return good

    # ---------------------------------------------------------------------
    # DETECT STOP SIGN
    # ---------------------------------------------------------------------
    '''
    def detect_stop_sign(self, cv_image):
        """
        Detectează semnul STOP utilizând culoarea roșie, forma octogonală și verificări suplimentare.
        """
        hsv = cv2.cvtColor(cv_image, cv2.COLOR_BGR2HSV)

        # Interval pentru roșu (ajustat pentru toleranță la marginea albă)
        lower_red1 = np.array([0, 50, 50])
        upper_red1 = np.array([10, 255, 255])
        lower_red2 = np.array([170, 50, 50])
        upper_red2 = np.array([180, 255, 255])

        mask1 = cv2.inRange(hsv, lower_red1, upper_red1)
        mask2 = cv2.inRange(hsv, lower_red2, upper_red2)
        red_mask = cv2.bitwise_or(mask1, mask2)

        # Morfologie pentru curățare
        kernel = np.ones((5, 5), np.uint8)
        red_mask = cv2.morphologyEx(red_mask, cv2.MORPH_OPEN, kernel)
        red_mask = cv2.morphologyEx(red_mask, cv2.MORPH_CLOSE, kernel)

        # Detectare contururi
        contours, _ = cv2.findContours(red_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        for cnt in contours:
            area = cv2.contourArea(cnt)
            if area < 100:  # Prag minim pentru dimensiunea semnului
                continue

            # Aproximare poligon
            approx = cv2.approxPolyDP(cnt, 0.03 * cv2.arcLength(cnt, True), True)
            sides = len(approx)

            # Verificăm dacă are formă octogonală
            if 6 <= sides <= 10:
                x, y, w, h = cv2.boundingRect(approx)
                aspect_ratio = w / float(h)

                # Validare pentru aspect ratio pătrat
                if 0.8 < aspect_ratio < 1.2:
                    # Desenăm contur pe imagine
                    cv2.drawContours(cv_image, [approx], -1, (0, 255, 255), 3)

                    # Opțional: verificare textură
                    roi = cv_image[y:y + h, x:x + w]
                    if self.validate_stop_sign_text(roi):
                        return True
        return False

    def validate_stop_sign_text(self, roi):
        """
        Validare suplimentară pentru semnul STOP prin textură (simplu sau OCR).
        """
        gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
        _, binary = cv2.threshold(gray, 100, 255, cv2.THRESH_BINARY)

        # Detectăm margini
        edges = cv2.Canny(binary, 50, 150)

        # Verificăm proporția pixelilor de margine
        non_zero_pixels = cv2.countNonZero(edges)
        total_pixels = roi.shape[0] * roi.shape[1]
        edge_density = non_zero_pixels / total_pixels

        # Dacă densitatea marginii e între un prag, e semn STOP
        return 0.02 < edge_density < 0.2
       '''

    # ---------------------------------------------------------------------
    # DETECT LINIE GALBENA
    # ---------------------------------------------------------------------
    def detect_yellow_line_center(self, cv_image):
        hsv = cv2.cvtColor(cv_image, cv2.COLOR_BGR2HSV)
        lower_y = np.array([20, 100, 100])
        upper_y = np.array([30, 255, 255])
        mask = cv2.inRange(hsv, lower_y, upper_y)
        kernel = np.ones((5,5), np.uint8)
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if not contours:
            return -1
        max_cnt = max(contours, key=cv2.contourArea)
        area = cv2.contourArea(max_cnt)
        if area < 50:
            return -1
        M = cv2.moments(max_cnt)
        if M["m00"] == 0:
            return -1
        cx = int(M["m10"] / M["m00"])
        return cx

    # ---------------------------------------------------------------------
    # STOP
    # ---------------------------------------------------------------------
    def stop_robot(self):
        twist = Twist()
        twist.linear.x = 0.0
        twist.angular.z = 0.0
        self.cmd_vel_pub.publish(twist)

    def show_debug(self, cv_image, edge):
        cv2.imshow("Camera", cv_image)
        if edge is not None:
            cv2.imshow("Edge", edge)
        cv2.waitKey(1)

def main(args=None):
    rclpy.init(args=args)
    node = LaneFollower()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()
        cv2.destroyAllWindows()

if __name__ == '__main__':
    main()
