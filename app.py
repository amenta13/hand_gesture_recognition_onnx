#!/usr/bin/env python

import csv
import copy
import argparse
import itertools
from math import degrees
from collections import Counter
from collections import deque

import cv2 as cv
import numpy as np
import pyautogui

from utils import CvFpsCalc
from utils.utils import rotate_and_crop_rectangle
from model import PalmDetection
from model import HandLandmark
from model import KeyPointClassifier
from model import PointHistoryClassifier
from collections import deque

def get_args():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        '-d',
        '--device',
        type=int,
        default=0,      # 0 = internal, 1 = external
    )
    parser.add_argument(
        '-im',
        '--image',
        type=str,
        default='',
    )
    parser.add_argument(
        '-wi',
        '--width',
        help='cap width',
        type=int,
        default=640,
    )
    parser.add_argument(
        '-he',
        '--height',
        help='cap height',
        type=int,
        default=480,
    )
    parser.add_argument(
        '-mdc',
        '--min_detection_confidence',
        help='min_detection_confidence',
        type=float,
        default=0.6,
    )
    parser.add_argument(
        '-dif',
        '--disable_image_flip',
        help='disable image flip',
        action='store_true',
    )


    args = parser.parse_args()

    return args


def main():
    # Argument Parsing
    args = get_args()

    if not args.image:
        cap_device = args.device
    else:
        cap_device = args.image
    cap_width = args.width
    cap_height = args.height
    min_detection_confidence = args.min_detection_confidence

    lines_hand = [
        [0,1],[1,2],[2,3],[3,4],
        [0,5],[5,6],[6,7],[7,8],
        [5,9],[9,10],[10,11],[11,12],
        [9,13],[13,14],[14,15],[15,16],
        [13,17],[17,18],[18,19],[19,20],[0,17],
    ]

    # Camera Setup
    cap = cv.VideoCapture(cap_device)
    cap.set(cv.CAP_PROP_FRAME_WIDTH, cap_width)
    cap.set(cv.CAP_PROP_FRAME_HEIGHT, cap_height)
    cap_fps = cap.get(cv.CAP_PROP_FPS)
    fourcc = cv.VideoWriter_fourcc('m', 'p', '4', 'v')
    video_writer = cv.VideoWriter(
        filename='output.mp4',
        fourcc=fourcc,
        fps=cap_fps,
        frameSize=(cap_width, cap_height),
    )

    # Model Loading
    palm_detection = PalmDetection(score_threshold=min_detection_confidence)
    hand_landmark = HandLandmark()

    keypoint_classifier = KeyPointClassifier()
    point_history_classifier = PointHistoryClassifier()

    # Load Labels
    with open(
        'model/keypoint_classifier/keypoint_classifier_label.csv',
        encoding='utf-8-sig',
    ) as f:
        keypoint_classifier_labels = csv.reader(f)
        keypoint_classifier_labels = [
            row[0] for row in keypoint_classifier_labels
        ]
    with open(
        'model/point_history_classifier/point_history_classifier_label.csv',
        encoding='utf-8-sig',
    ) as f:
        point_history_classifier_labels = csv.reader(f)
        point_history_classifier_labels = [
            row[0] for row in point_history_classifier_labels
        ]

    # FPS Measurement Module
    cvFpsCalc = CvFpsCalc(buffer_len=10)

    # Coordinate History
    history_length = 16
    # point_history = deque(maxlen=history_length)
    point_history = {}
    pre_point_history = {}

    # Finger Gesture History
    gesture_history_length = 10
    finger_gesture_history = {}

    # Latest History of Palm Center Coordinates for Hand Tracking
    # {
    #   int(trackid1): [cx, cy],
    #   int(trackid2): [cx, cy],
    #   int(trackid3): [cx, cy],
    #     :
    # }
    palm_trackid_cxcy = {}

    mode = 0
    wh_ratio = cap_width / cap_height

    auto = False
    prev_number = -1
    image = None

    import time
    last_click_time = 0
    CLICK_DELAY = 0.5  # seconds

    finger_trail = deque(maxlen=500)

    while True:
        fps = cvFpsCalc.get()

        # Key Handling (ESC: Exit)
        key = cv.waitKey(1) if not args.image else cv.waitKey(0) if image is not None and args.image else cv.waitKey(1)
        if key == ord('s'):  # S to screenshot
            screenshot = pyautogui.screenshot()
            screenshot.save("screenshot.png")
            print("Screenshot saved")
        if key == 27:  # ESC
            break
        number, mode, auto, prev_number = select_mode(key, mode, auto, prev_number)

        # Camera Capture
        ret, image = cap.read()
        if not ret:
            break
        image = image if args.disable_image_flip else cv.flip(image, 1) # ミラー表示

        # Crop and zoom
        # Adjustable zoom factor (1.0 = no zoom, 1.5 = 1.5× zoom, 2.0 = 2× zoom)
        ZOOM = 1.9

        # Compute crop size
        crop_w = int(cap_width / ZOOM)
        crop_h = int(cap_height / ZOOM)

        # Center crop
        center_x = cap_width // 2
        center_y = cap_height // 2

        TABLE_X1 = center_x - crop_w // 2
        TABLE_Y1 = center_y - crop_h // 2
        TABLE_X2 = center_x + crop_w // 2
        TABLE_Y2 = center_y + crop_h // 2

        # Ensure bounds
        TABLE_X1 = max(0, TABLE_X1)
        TABLE_Y1 = max(0, TABLE_Y1)
        TABLE_X2 = min(cap_width, TABLE_X2)
        TABLE_Y2 = min(cap_height, TABLE_Y2)

        # Crop to table
        table_region = image[TABLE_Y1:TABLE_Y2, TABLE_X1:TABLE_X2]

        # Zoom to full screen size (same size as cap_width × cap_height)
        image = cv.resize(table_region, (cap_width, cap_height), interpolation=cv.INTER_LINEAR)

        debug_image = copy.deepcopy(image)

        # Detection Execution

        # ============================================================= PalmDetection
        # Hand Detection - Single-Batch Processing
        hands = palm_detection(image)
        # hand: sqn_rr_size, rotation, sqn_rr_center_x, sqn_rr_center_y

        rects = []
        not_rotate_rects = []
        rects_tuple = None
        cropted_rotated_hands_images = []

        # If the number of detected hands drops to zero, reset the latest history of the tracking palm center coordinates.
        if len(hands) == 0:
            palm_trackid_cxcy = {}
        # List Linking the Latest History of Palm Center Coordinates for Tracking to Bounding Box Detection Order
        palm_trackid_box_x1y1s = {}

        if len(hands) > 0:
            for hand in hands:
                # hand: sqn_rr_size, rotation, sqn_rr_center_x, sqn_rr_center_y
                sqn_rr_size = hand[0]
                rotation = hand[1]
                sqn_rr_center_x = hand[2]
                sqn_rr_center_y = hand[3]

                cx = int(sqn_rr_center_x * cap_width)
                cy = int(sqn_rr_center_y * cap_height)
                xmin = int((sqn_rr_center_x - (sqn_rr_size / 2)) * cap_width)
                xmax = int((sqn_rr_center_x + (sqn_rr_size / 2)) * cap_width)
                ymin = int((sqn_rr_center_y - (sqn_rr_size * wh_ratio / 2)) * cap_height)
                ymax = int((sqn_rr_center_y + (sqn_rr_size * wh_ratio / 2)) * cap_height)
                xmin = max(0, xmin)
                xmax = min(cap_width, xmax)
                ymin = max(0, ymin)
                ymax = min(cap_height, ymax)
                degree = degrees(rotation)
                # [boxcount, cx, cy, width, height, degree]
                rects.append([cx, cy, (xmax-xmin), (ymax-ymin), degree])

            rects = np.asarray(rects, dtype=np.float32)

            # Acquisition of a palm image with its rotation angle corrected to zero degrees
            cropted_rotated_hands_images = rotate_and_crop_rectangle(
                image=image,
                rects_tmp=rects,
                operation_when_cropping_out_of_range='padding',
            )

            # Debug ===============================================================
            for rect in rects:
                # Rendering of Region with Rotation, Red Frame
                rects_tuple = ((rect[0], rect[1]), (rect[2], rect[3]), rect[4])
                box = cv.boxPoints(rects_tuple).astype(np.intp)
                # cv.drawContours(debug_image, [box], 0,(0,0,255), 2, cv.LINE_AA)       # Shows bounding box for debugging purposes

                # Rendering of Region Ignoring Rotation, Orange Frame
                rcx = int(rect[0])
                rcy = int(rect[1])
                half_w = int(rect[2] // 2)
                half_h = int(rect[3] // 2)
                x1 = rcx - half_w
                y1 = rcy - half_h
                x2 = rcx + half_w
                y2 = rcy + half_h
                text_x = max(x1, 10)
                text_x = min(text_x, cap_width-120)
                text_y = max(y1-15, 45)
                text_y = min(text_y, cap_height-20)
                # [boxcount, rcx, rcy, x1, y1, x2, y2, height, degree]
                not_rotate_rects.append([rcx, rcy, x1, y1, x2, y2, 0])
                # Detection Box Size WxH
                """
                cv.putText(
                    debug_image,
                    f'{y2-y1}x{x2-x1}',
                    (text_x, text_y),
                    cv.FONT_HERSHEY_SIMPLEX,
                    0.8,
                    (0,0,0),
                    2,
                    cv.LINE_AA,
                )
                cv.putText(
                    debug_image,
                    f'{y2-y1}x{x2-x1}',
                    (text_x, text_y),
                    cv.FONT_HERSHEY_SIMPLEX,
                    0.8,
                    (59,255,255),
                    1,
                    cv.LINE_AA,
                )
                """
                # Drawing Detection Box
                """
                cv.rectangle(
                    debug_image,
                    (x1,y1),
                    (x2,y2),
                    (0,128,255),
                    2,
                    cv.LINE_AA,
                )
                """
                # Drawing the Center Coordinates of the Detection Region
                cv.circle(
                    debug_image,
                    (rcx, rcy),
                    3,
                    (0, 255, 255),
                    -1,
                )
                """
                Saving the Latest History of Palm Center Coordinates for Palm Tracking
                    1. Extract the central coordinates closest to the reference point from the historical data.
                    2. If the distance exceeds 100px, recognize it as a new palm.
                    3. If the distance is 100px or less, assign the corresponding track ID and overwrite the center coordinates in the historical data.
                """
                # 1. Extract the central coordinates closest to the reference point from the historical data.
                base_point = np.asarray(
                    [rcx, rcy],
                    dtype=np.float32,
                )
                points = np.asarray(
                    list(palm_trackid_cxcy.values()),
                    dtype=np.float32,
                )
                if len(points) > 0:
                    # Nearest Neighbor Search
                    diff_val = points - base_point
                    all_points_distance = np.linalg.norm(diff_val, axis=1)
                    nearest_trackid = np.argmin(all_points_distance)
                    nearest_distance = all_points_distance[nearest_trackid]
                    new_trackid = int(nearest_trackid) + 1
                    # 2. If the distance exceeds 100px, recognize it as a new palm.
                    # 3. If the distance is 100px or less, assign the corresponding track ID and overwrite the center coordinates in the historical data.
                    if nearest_distance > 100:
                        # Generate a new track ID by adding 1 to the current maximum track ID.
                        new_trackid = next(iter(reversed(palm_trackid_cxcy))) + 1
                else:
                    # Initial value of trackid
                    new_trackid = 1

                # Update or add a new entry to the latest history of palm center coordinates for palm tracking.
                palm_trackid_cxcy[new_trackid] = [rcx, rcy]
                # Align the bounding box detection order with the trackid order.
                # box_x1y1x2y2_palm_trackids.append([x1, y1, x2, y2, new_trackid])
                palm_trackid_box_x1y1s[new_trackid] = [x1, y1]
                # Debug ===============================================================

        # ============================================================= HandLandmark
        if len(cropted_rotated_hands_images) > 0:

            # Inference HandLandmark - Batch Processing
            hand_landmarks, rotated_image_size_leftrights = hand_landmark(
                images=cropted_rotated_hands_images,
                rects=rects,
            )

            if len(hand_landmarks) > 0:
                # Draw
                pre_processed_landmarks = []
                pre_processed_point_histories = []
                for (trackid, x1y1), landmark, rotated_image_size_leftright, not_rotate_rect in \
                    zip(palm_trackid_box_x1y1s.items(), hand_landmarks, rotated_image_size_leftrights, not_rotate_rects):

                    x1, y1 = x1y1
                    rotated_image_width, _, left_hand_0_or_right_hand_1 = rotated_image_size_leftright
                    thick_coef = rotated_image_width / 400
                    lines = np.asarray(
                        [
                            np.array([landmark[point] for point in line]).astype(np.int32) for line in lines_hand
                        ]
                    )
                    radius = int(1+thick_coef*5)
                    cv.polylines(
                        debug_image,
                        lines,
                        False,
                        (255, 0, 0),
                        int(radius),
                        cv.LINE_AA,
                    )
                    _ = [cv.circle(debug_image, (int(x), int(y)), radius, (0,128,255), -1) for x,y in landmark[:,:2]]
                    left_hand_0_or_right_hand_1 = left_hand_0_or_right_hand_1 if args.disable_image_flip else (1 - left_hand_0_or_right_hand_1)
                    handedness = 'Left ' if left_hand_0_or_right_hand_1 == 0 else 'Right'
                    _, _, x1, y1, _, _, _ = not_rotate_rect
                    text_x = max(x1, 10)
                    text_x = min(text_x, cap_width-120)
                    text_y = max(y1-70, 20)
                    text_y = min(text_y, cap_height-70)
                    cv.putText(
                        debug_image,
                        f'trackid:{trackid} {handedness}',
                        (text_x, text_y),
                        cv.FONT_HERSHEY_SIMPLEX,
                        0.8,
                        (0,0,0),
                        2,
                        cv.LINE_AA,
                    )
                    cv.putText(
                        debug_image,
                        f'trackid:{trackid} {handedness}',
                        (text_x, text_y),
                        cv.FONT_HERSHEY_SIMPLEX,
                        0.8,
                        (59,255,255),
                        1,
                        cv.LINE_AA,
                    )

                    # Conversion to Relative and Normalized Coordinates
                    """
                    pre_processed_landmark: np.ndarray [42], [x,y]x21
                    """
                    pre_processed_landmark = pre_process_landmark(
                        landmark,
                    )
                    pre_processed_landmarks.append(pre_processed_landmark)

                """
                point_history: dict
                {
                    int(trackid1): [[x, y],[x, y],[x, y],[x, y], ...],
                    int(trackid2): [[x, y],[x, y], ...],
                    int(trackid3): [[x, y],[x, y],[x, y], ...],
                        :
                }
                    ↓
                pre_processed_point_histories: List
                [
                    [rx, ry, rx, ry, rx, ry, rx, ry, ...],
                    [rx, ry, rx, ry, ...],
                    [rx, ry, rx, ry, rx, ry, ...],
                        :
                ]
                """
                # Convert index finger trajectory to relative coordinates.
                pre_processed_point_histories = pre_process_point_history(
                    image_width=debug_image.shape[1],
                    image_height=debug_image.shape[0],
                    point_history=point_history,
                )

                # Save Training Data
                logging_csv(
                    number,
                    mode,
                    trackid,
                    pre_processed_landmark,
                    pre_processed_point_histories,
                )

                # Hand Sign Classification – Batch Processing
                hand_sign_ids = keypoint_classifier(
                    np.asarray(pre_processed_landmarks, dtype=np.float32)
                )
                for (trackid, x1y1), landmark, hand_sign_id in zip(palm_trackid_box_x1y1s.items(), hand_landmarks, hand_sign_ids):
                    x1, y1 = x1y1

                    # Compute per-hand text anchor
                    text_x = max(x1, 10)
                    text_x = min(text_x, cap_width - 120)
                    text_y = max(y1 - 70, 20)
                    text_y = min(text_y, cap_height - 70)

                    point_history.setdefault(trackid, deque(maxlen=history_length))
                    if hand_sign_id == 2:  # Pointing Gesture
                        point_history[trackid].append(list(landmark[8])) # Index Finger Coordinates
                    else:
                        point_history[trackid].append([0, 0])

                    if hand_sign_id < len(keypoint_classifier_labels):
                        hand_pose_label = keypoint_classifier_labels[hand_sign_id]
                    else:
                        hand_pose_label = "None"

                    # --- Mouse Control ----------------------------------------------------
                    # Only click if pointing
                    if hand_sign_id == 2:  # pointing gesture

                        # Trailing on pointing fingertip
                        """
                        fx = int(landmark[8][0])
                        fy = int(landmark[8][1])
                        finger_trail.append((fx, fy))
                        """

                        current_time = time.time()
                        if current_time - last_click_time > CLICK_DELAY:

                            # Index finger tip
                            finger_x = int(landmark[8][0])
                            finger_y = int(landmark[8][1])

                            # Convert to screen coordinates
                            screen_w, screen_h = pyautogui.size()
                            screen_x = int((finger_x / cap_width) * screen_w)
                            screen_y = int((finger_y / cap_height) * screen_h)

                            # Move mouse and click
                            pyautogui.moveTo(screen_x, screen_y)
                            pyautogui.click()
                            last_click_time = current_time

                    # ----------------------------------------------------------------------

                    cv.putText(
                        debug_image,
                        f'Pose: {hand_pose_label}',
                        (text_x, text_y + 25),
                        cv.FONT_HERSHEY_SIMPLEX,
                        0.8,
                        (0,0,0),
                        2,
                        cv.LINE_AA,
                    )
                    cv.putText(
                        debug_image,
                        f'Pose: {hand_pose_label}',
                        (text_x, text_y + 25),
                        cv.FONT_HERSHEY_SIMPLEX,
                        0.8,
                        (59,255,255),
                        1,
                        cv.LINE_AA,
                    )

                # Draw fingertip trail
                """
                for i in range(1, len(finger_trail)):
                    cv.line(
                        debug_image,
                        finger_trail[i - 1],
                        finger_trail[i],
                        (160, 32, 240),  # bright yellow
                        2,
                        cv.LINE_AA
                    )
                """


                """
                To avoid the trajectory of the index finger remaining visible on the display
                If there is a palm that is no longer being tracked (i.e., has moved out of the field of view), clear the history of the
                index finger's X/Y coordinates.
                For any TrackID where the complete set of current trajectory coordinates matches the complete set of previous trajectory
                coordinates exactly, the history information is deemed unchanged and is subsequently removed from the history.
                point_history: 16 Latest Trajectories
                pre_point_history: Previous Trajectory: 16 Points
                """
                if len(pre_point_history) > 0:
                    temp_point_history = copy.deepcopy(point_history)
                    for track_id, points in temp_point_history.items():
                        if track_id in pre_point_history:
                            pre_points = pre_point_history[track_id]
                            if points == pre_points:
                                _ = point_history.pop(track_id, None)
                pre_point_history = copy.deepcopy(point_history)

                # Finger Gesture Classification – Batch Processing
                finger_gesture_ids = None
                temp_trackid_x1y1s = {}
                temp_pre_processed_point_history = []
                for (trackid, x1y1), pre_processed_point_history in zip(palm_trackid_box_x1y1s.items(), pre_processed_point_histories):
                    point_history_len = len(pre_processed_point_history)
                    if point_history_len > 0 and point_history_len % (history_length * 2) == 0:
                        temp_trackid_x1y1s[trackid] = x1y1
                        temp_pre_processed_point_history.append(pre_processed_point_history)
                if len(temp_pre_processed_point_history) > 0:
                    finger_gesture_ids = point_history_classifier(
                        temp_pre_processed_point_history,
                    )

                # Calculate the most frequent Gesture ID among the most recently detected gestures.
                if finger_gesture_ids is not None:
                    for (trackid, x1y1), finger_gesture_id in zip(temp_trackid_x1y1s.items(), finger_gesture_ids):
                        x1, y1 = x1y1
                        trackid_str = str(trackid)
                        finger_gesture_history.setdefault(trackid_str, deque(maxlen=gesture_history_length))
                        finger_gesture_history[trackid_str].append(int(finger_gesture_id))
                        most_common_fg_id = Counter(finger_gesture_history[trackid_str]).most_common()
                        text_x = max(x1, 10)
                        text_x = min(text_x, cap_width-120)
                        text_y = max(y1-45, 20)
                        text_y = min(text_y, cap_height-45)
                        classifier_label = point_history_classifier_labels[most_common_fg_id[0][0]]
                        # print(f'trackid: {trackid} [x1,y1]: [{x1},{y1}] finger_gesture_id: {classifier_label}')
                        """
                        cv.putText(
                            debug_image,
                            f'{classifier_label}',
                            (text_x, text_y+50),
                            cv.FONT_HERSHEY_SIMPLEX,
                            0.8,
                            (0,0,0),
                            2,
                            cv.LINE_AA,
                        )
                        cv.putText(
                            debug_image,
                            f'{classifier_label}',
                            (text_x, text_y+50),
                            cv.FONT_HERSHEY_SIMPLEX,
                            0.8,
                            (59,255,255),
                            1,
                            cv.LINE_AA,
                        )
                        """

            else:
                point_history = {}

        else:
            point_history = {}

        debug_image = draw_point_history(debug_image, point_history)
        debug_image = draw_info(debug_image, fps, mode, number, auto)

        # Change to True to hide display window
        hidden = False

        if not hidden:
            # Screen reflection
            cv.namedWindow('Hand Gesture Recognition', cv.WND_PROP_FULLSCREEN)
            cv.setWindowProperty('Hand Gesture Recognition', cv.WND_PROP_FULLSCREEN, cv.WINDOW_FULLSCREEN)
            cv.imshow('Hand Gesture Recognition', debug_image)
            video_writer.write(debug_image)
        else:
            # Hidden window so cv.waitKey still receives keyboard input
            cv.namedWindow('Hidden', cv.WINDOW_NORMAL)
            cv.setWindowProperty('Hidden', cv.WND_PROP_VISIBLE, 0)


    if video_writer:
        video_writer.release()
    if cap:
        cap.release()
    cv.destroyAllWindows()


def select_mode(key, mode, auto=False, prev_number=-1):
    number = -1
    if 48 <= key <= 57:  # 0 ~ 9
        number = key - 48
        prev_number = number
    if key == 110:  # n
        mode = 0
    if key == 107:  # k
        mode = 1
    if key == 104:  # h
        mode = 2
    if key == 97:   # a
        auto = not auto
    if auto == True:
        if prev_number != -1:
            number = prev_number
    else:
        prev_number = -1

    return number, mode, auto, prev_number


def pre_process_landmark(landmark_list):
    if len(landmark_list) == 0:
        return []

    temp_landmark_list = copy.deepcopy(landmark_list)
    # Convert to Relative Coordinates
    base_x, base_y = temp_landmark_list[0][0], temp_landmark_list[0][1]
    temp_landmark_list = [
        [temp_landmark[0] - base_x, temp_landmark[1] - base_y] for temp_landmark in temp_landmark_list
    ]
    # Convert to a 1-dimensional list
    temp_landmark_list = list(
        itertools.chain.from_iterable(temp_landmark_list)
    )
    # Normalization
    max_value = max(list(map(abs, temp_landmark_list)))

    def normalize_(n):
        return n / max_value

    temp_landmark_list = list(map(normalize_, temp_landmark_list))
    return temp_landmark_list


def pre_process_point_history(
    image_width: int,
    image_height: int,
    point_history: dict,
):
    """pre_process_point_history

    Parameters
    ----------
    image_width: int
        Input image width

    image_height: int
        Input image height

    point_history: dict
        Index finger XY coordinate history per trackid (detected palm)
        {
            int(trackid1): [[x, y],[x, y],[x, y],[x, y], ...],
            int(trackid2): [[x, y],[x, y], ...],
            int(trackid3): [[x, y],[x, y],[x, y], ...],
                :
        }

    Returns
    -------
    relative_coordinate_list_by_trackid: List
        [
            [rx, ry, rx, ry, rx, ry, rx, ry, ...],
            [rx, ry, rx, ry, ...],
            [rx, ry, rx, ry, rx, ry, ...],
                :
        ]
    """
    if len(point_history) == 0:
        return []

    temp_point_history = copy.deepcopy(point_history)
    relative_coordinate_list_by_trackid = []

    # Convert to relative coordinates per track ID.
    for trackid, points in temp_point_history.items():
        base_x, base_y = points[0][0], points[0][1]
        relative_coordinate_list = [
            [
                (point[0] - base_x) / image_width,
                (point[1] - base_y) / image_height,
            ] for point in points
        ]
        # Convert to a 1-dimensional list
        relative_coordinate_list_1d = list(
            itertools.chain.from_iterable(relative_coordinate_list)
        )
        relative_coordinate_list_by_trackid.append(relative_coordinate_list_1d)
    return relative_coordinate_list_by_trackid


def logging_csv(number, mode, trackid, landmark_list, point_histories):
    if mode == 0:
        pass
    if mode == 1 and (0 <= number <= 9):
        csv_path = 'model/keypoint_classifier/keypoint.csv'
        with open(csv_path, 'a', newline="") as f:
            writer = csv.writer(f)
            writer.writerow([number, trackid, *landmark_list])
    if mode == 2 and (0 <= number <= 9):
        csv_path = 'model/point_history_classifier/point_history.csv'
        with open(csv_path, 'a', newline="") as f:
            writer = csv.writer(f)
            for point_history in point_histories:
                writer.writerow([number, trackid, *point_history])


def draw_info_text(
    image,
    brect,
    handedness,
    hand_sign_text,
    finger_gesture_text
):
    info_text = handedness
    if hand_sign_text != "":
        info_text = f'{handedness}:{hand_sign_text}'
    cv.putText(
        image,
        info_text,
        (brect[0] + 5, brect[1] - 4),
        cv.FONT_HERSHEY_SIMPLEX,
        0.6,
        (255, 255, 255),
        1,
        cv.LINE_AA,
    )

    if finger_gesture_text != "":
        cv.putText(
            image,
            f'Finger Gesture:{finger_gesture_text}',
            (10, 60),
            cv.FONT_HERSHEY_SIMPLEX,
            1.0,
            (0, 0, 0),
            4,
            cv.LINE_AA,
        )
        cv.putText(
            image,
            f'Finger Gesture:{finger_gesture_text}',
            (10, 60),
            cv.FONT_HERSHEY_SIMPLEX,
            1.0,
            (255, 255, 255),
            2,
            cv.LINE_AA,
        )

    return image


def draw_point_history(image, point_history):
    _ = [
        cv.circle(image, (point[0], point[1]), 1 + int(index / 2), (152, 251, 152), 2) \
            for trackid, points in point_history.items() \
                for index, point in enumerate(points) if point[0] != 0 and point[1] != 0
    ]
    return image


def draw_info(image, fps, mode, number, auto):
    cv.putText(
        image,
        f'FPS:{str(fps)}',
        (10, 30),
        cv.FONT_HERSHEY_SIMPLEX,
        1.0,
        (0, 0, 0),
        4,
        cv.LINE_AA,
    )
    cv.putText(
        image,
        f'FPS:{str(fps)}',
        (10, 30),
        cv.FONT_HERSHEY_SIMPLEX,
        1.0,
        (255, 255, 255),
        2,
        cv.LINE_AA,
    )

    mode_string = ['Logging Key Point', 'Logging Point History']
    if 1 <= mode <= 2:
        cv.putText(
            image,
            f'MODE:{mode_string[mode - 1]}',
            (10, 90),
            cv.FONT_HERSHEY_SIMPLEX,
            0.6,
            (255, 255, 255),
            1,
            cv.LINE_AA,
        )
        if 0 <= number <= 9:
            cv.putText(
                image,
                f'NUM:{str(number)}',
                (10, 110),
                cv.FONT_HERSHEY_SIMPLEX,
                0.6,
                (255, 255, 255),
                1,
                cv.LINE_AA,
            )
    """
    cv.putText(
        image,
        f'AUTO:{str(auto)}',
        (10, 130),
        cv.FONT_HERSHEY_SIMPLEX,
        0.6,
        (255, 255, 255),
        1,
        cv.LINE_AA,
    )
    """
    return image


if __name__ == '__main__':
    main()
