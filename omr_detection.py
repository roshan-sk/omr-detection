import cv2
import numpy as np


def get_contours(edged):
    contours, _ = cv2.findContours(edged, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
    contours = sorted(contours, key=cv2.contourArea, reverse=True)

    for cnt in contours:
        if cv2.contourArea(cnt) < 100000:
            continue
        peri = cv2.arcLength(cnt, True)
        approx = cv2.approxPolyDP(cnt, 0.02 * peri, True)
        if len(approx) == 4:
            return approx
    return None


def reorder(points):
    points = points.reshape((4, 2))
    new_points = np.zeros((4, 1, 2), dtype=np.int32)

    add = points.sum(1)
    new_points[0] = points[np.argmin(add)]
    new_points[3] = points[np.argmax(add)]

    diff = np.diff(points, axis=1)
    new_points[1] = points[np.argmin(diff)]
    new_points[2] = points[np.argmax(diff)]

    return new_points


def warp_image(image, points):
    points = reorder(points)
    pts1 = np.float32(points)
    pts2 = np.float32([[0, 0], [800, 0], [0, 1200], [800, 1200]])

    matrix = cv2.getPerspectiveTransform(pts1, pts2)
    return cv2.warpPerspective(image, matrix, (800, 1200))


def preprocess_and_warp(image):
    image = cv2.resize(image, (800, 1200))
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(gray, (5, 5), 0)
    edged = cv2.Canny(blur, 50, 150)

    contour = get_contours(edged)
    if contour is None or cv2.contourArea(contour) < 300000:
        return image

    return warp_image(image, contour)

def get_thresh(img):
    if len(img.shape) == 2:  # already gray
        gray = img
    else:
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    thresh = cv2.adaptiveThreshold(
        gray, 255,
        cv2.ADAPTIVE_THRESH_MEAN_C,
        cv2.THRESH_BINARY_INV,
        21, 5
    )
    return gray, thresh


def threshold_column(img):
    _, thresh = get_thresh(img)
    return thresh


def get_row_centers(rows):
    return [(s + e) // 2 for (s, e) in rows]


def detect_rows(img, min_h=8, th_start=0.17, th_end=0.2):
    # gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    # thresh = cv2.adaptiveThreshold(
    #     gray, 255,
    #     cv2.ADAPTIVE_THRESH_MEAN_C,
    #     cv2.THRESH_BINARY_INV,
    #     21, 5
    # )
    _, thresh = get_thresh(img)

    row_sum = np.sum(thresh, axis=1)
    if np.max(row_sum) == 0:
        return []

    row_sum = row_sum / np.max(row_sum)

    rows, in_row = [], False

    for i in range(len(row_sum)):
        if row_sum[i] > th_start and not in_row:
            start = i
            in_row = True
        elif row_sum[i] < th_end and in_row:
            rows.append((start, i))
            in_row = False

    return [(s, e) for (s, e) in rows if (e - s) > min_h]


def crop_name_area(image):
    h, w = image.shape[:2]
    return image[
        int(h * 0.17):int(h * 0.523),
        int(w * 0.081):w - int(w * 0.44)
    ]


def split_name_columns(img, n=25):
    h, w = img.shape[:2]
    return [img[int(0):h, int(i*w/n):int((i+1)*w/n)] for i in range(n)]


def detect_name(img):
    cols = split_name_columns(img, 25)
    rows = detect_rows(img)
    centers = get_row_centers(rows)

    name = ""

    for col in cols:
        thresh = threshold_column(col)
        scores = []

        for c in centers:
            top = max(0, c - 8)
            bottom = min(col.shape[0], c + 8)

            h, w = thresh.shape
            bubble = thresh[top:bottom, int(w*0.25):int(w*0.75)]
            scores.append(cv2.countNonZero(bubble))

        s = sorted(scores, reverse=True)
        if s[0] < 60 or (s[0] - s[1]) < 15:
            name += " "
        else:
            name += chr(ord('A') + np.argmax(scores))

    return name.strip()


def crop_class_area(image):
    h, w = image.shape[:2]
    return image[
        int(h * 0.18):int(h * 0.28),
        int(w * 0.615):int(w * 0.66)
    ]


def detect_class(img):
    thresh = threshold_column(img)
    rows = detect_rows(img)
    centers = get_row_centers(rows)

    scores = []

    for c in centers:
        top = max(0, c - 5)
        bottom = min(img.shape[0], c + 5)

        h, w = thresh.shape
        bubble = thresh[top:bottom, int(w*0.55):int(w*0.85)]
        scores.append(cv2.countNonZero(bubble))

    s = sorted(scores, reverse=True)
    if s[0] < 50 or (s[0] - s[1]) < 10:
        return ""

    return ["6th", "7th", "8th", "9th", "10th", "11th", "12th"][np.argmax(scores)]

def crop_section_area(image):
    h, w = image.shape[:2]

    return image[
        int(h * 0.17):int(h * 0.305),
        int(w * 0.69):int(w * 0.725)
    ]
    
def detect_section(img):
    thresh = threshold_column(img)
    rows = detect_rows(img)
    centers = get_row_centers(rows)

    scores = []

    for c in centers:
        top = max(0, c - 5)
        bottom = min(img.shape[0], c + 5)

        h, w = thresh.shape
        bubble = thresh[top:bottom, int(w*0.55):int(w*0.85)]

        scores.append(cv2.countNonZero(bubble))

    s = sorted(scores, reverse=True)

    if s[0] < 50 or (s[0] - s[1]) < 10:
        return ""

    return chr(ord('A') + np.argmax(scores))

def crop_roll_area(image):
    h, w = image.shape[:2]

    return image[
        int(h * 0.17):int(h * 0.308),
        int(w * 0.737):int(w * 0.79)
    ]

def split_roll_columns(img, n=2):
    h, w = img.shape[:2]
    return [img[:, int(i*w/n):int((i+1)*w/n)] for i in range(n)]

def detect_roll_number(img, num_digits=2):
    cols = split_roll_columns(img, num_digits)

    rows = detect_rows(img)
    centers = get_row_centers(rows)

    roll = ""

    for col in cols:
        thresh = threshold_column(col)
        scores = []

        for c in centers:
            top = max(0, c - 6)
            bottom = min(col.shape[0], c + 6)

            h, w = thresh.shape
            bubble = thresh[top:bottom, int(w*0.25):int(w*0.75)]

            scores.append(cv2.countNonZero(bubble))

        s = sorted(scores, reverse=True)

        if s[0] < 50 or (s[0] - s[1]) < 10:
            roll += ""
        else:
            digit = str(np.argmax(scores))
            roll += digit

    return roll

def crop_stream_area(image):
    h, w = image.shape[:2]

    return image[
        int(h * 0.17):int(h * 0.225),
        int(w * 0.87):int(w * 0.92)
    ]
    
def detect_stream(img):
    thresh = threshold_column(img)
    rows = detect_rows(img)
    centers = get_row_centers(rows)

    scores = []

    for c in centers:
        top = max(0, c - 5)
        bottom = min(img.shape[0], c + 5)

        h, w = thresh.shape
        bubble = thresh[top:bottom, int(w*0.5):int(w*0.9)]

        scores.append(cv2.countNonZero(bubble))

    s = sorted(scores, reverse=True)

    if s[0] < 50 or (s[0] - s[1]) < 10:
        return ""

    streams = ["PCM", "PCB", "COM", "HUM"]

    return streams[np.argmax(scores)]

def crop_set_area(image):
    h, w = image.shape[:2]

    return image[
        int(h * 0.265):int(h * 0.305),
        int(w * 0.87):int(w * 0.92)
    ]
    
def detect_set(img):
    thresh = threshold_column(img)
    rows = detect_rows(img)
    centers = get_row_centers(rows)

    scores = []

    for c in centers:
        top = max(0, c - 5)
        bottom = min(img.shape[0], c + 5)

        h, w = thresh.shape
        bubble = thresh[top:bottom, int(w*0.5):int(w*0.9)]

        scores.append(cv2.countNonZero(bubble))

    s = sorted(scores, reverse=True)

    if len(s) < 2 or s[0] < 50 or (s[0] - s[1]) < 10:
        return ""

    return str(np.argmax(scores) + 1)


def crop_subject_code_area(image):
    h, w = image.shape[:2]

    return image[
        int(h * 0.385):int(h * 0.522),
        int(w * 0.58):int(w * 0.74)
    ]

def split_subject_columns(img, n=3):
    h, w = img.shape[:2]
    return [img[:, int(i*w/n):int((i+1)*w/n)] for i in range(n)]

def detect_subject_code(img, num_digits=3):
    cols = split_subject_columns(img, num_digits)

    rows = detect_rows(img)
    centers = get_row_centers(rows)

    code = ""

    for col in cols:
        thresh = threshold_column(col)
        scores = []

        for c in centers:
            top = max(0, c - 6)
            bottom = min(col.shape[0], c + 6)

            h, w = thresh.shape
            bubble = thresh[top:bottom, int(w*0.25):int(w*0.75)]

            scores.append(cv2.countNonZero(bubble))

        s = sorted(scores, reverse=True)

        if s[0] < 50 or (s[0] - s[1]) < 10:
            code += ""
        else:
            digit = str(np.argmax(scores))
            code += digit

    return code


def crop_admission_area(image):
    h, w = image.shape[:2]

    return image[
        int(h * 0.385):int(h * 0.522),
        int(w * 0.76):int(w * 0.92)
    ]
    
    
def split_admission_columns(img, n=5):
    h, w = img.shape[:2]
    return [img[:, int(i*w/n):int((i+1)*w/n)] for i in range(n)]


def detect_admission_number(img, num_digits=5):
    cols = split_admission_columns(img, num_digits)

    rows = detect_rows(img)
    centers = get_row_centers(rows)

    admission = ""

    for col in cols:
        thresh = threshold_column(col)
        scores = []

        for c in centers:
            top = max(0, c - 6)
            bottom = min(col.shape[0], c + 6)

            h, w = thresh.shape
            bubble = thresh[top:bottom, int(w*0.25):int(w*0.75)]

            scores.append(cv2.countNonZero(bubble))

        s = sorted(scores, reverse=True)

        if len(s) < 2:
            admission += ""
            continue

        if s[0] < 50 or (s[0] - s[1]) < 10:
            admission += ""
        else:
            digit = str(np.argmax(scores))
            admission += digit

    return admission    
    
    
def crop_answer_area(image):
    h, w = image.shape[:2]

    return image[
        int(h * 0.535):int(h * 0.875),
        int(w * 0.07):int(w * 0.71)
    ]
    
def crop_block_inner(block):
    h, w = block.shape[:2]

    return block[
        int(h * 0.02):int(h * 1.2),
        int(w * 0.20):int(w * 0.975)
    ]
    
    
def split_answer_blocks(img, num_blocks=4):
    h, w = img.shape[:2]

    blocks = []

    for i in range(num_blocks):
        x1 = int(i * w / num_blocks)
        x2 = int((i + 1) * w / num_blocks)

        block = img[:, x1:x2]

        block = crop_block_inner(block)

        blocks.append(block)

    return blocks


def detect_answers_block(block_img, centers, full_h):
    thresh = threshold_column(block_img)

    answers = []

    scale = block_img.shape[0] / full_h

    for c in centers:
        c_scaled = int(c * scale)

        top = max(0, c_scaled - 7)
        bottom = min(block_img.shape[0], c_scaled + 7)

        row_img = thresh[top:bottom, :]

        h, w = row_img.shape
        option_width = w // 4

        scores = []

        for i in range(4):
            x1 = i * option_width
            x2 = (i + 1) * option_width

            option = row_img[
                int(h * 0.2):int(h * 0.8),
                x1 + int(option_width * 0.25):x2 - int(option_width * 0.25)
            ]

            total_pixels = option.shape[0] * option.shape[1]
            filled = cv2.countNonZero(option)
            scores.append(filled / total_pixels)
            

        sorted_scores = sorted(scores, reverse=True)

        max_val = sorted_scores[0]
        second_val = sorted_scores[1]
        third_val = sorted_scores[2]

        diff12 = max_val - second_val
        diff23 = second_val - third_val

        if max_val < 0.28:
            answers.append("-")
            
        elif diff12 < 0.08 and diff23 < 0.08:
            answers.append("-")

        elif (
            max_val > 0.42 and
            (second_val / max_val) > 0.80 and
            diff12 < 0.10 and
            diff23 > 0.10                 
        ):
            answers.append("MULTI")

        elif (
            max_val > 0.34 and
            diff12 > 0.10 and           
            second_val < (max_val * 0.92)
        ):
            answers.append(["A", "B", "C", "D"][np.argmax(scores)])

        else:
            answers.append("-")

    return answers


def detect_all_answers(answer_img):
    blocks = split_answer_blocks(answer_img, 4)

    rows = detect_rows(answer_img)
    centers = get_row_centers(rows)

    all_answers = []

    for block in blocks:
        answers = detect_answers_block(block, centers, answer_img.shape[0])
        all_answers.extend(answers)

    return all_answers


    

def process_omr(image):

    warped = preprocess_and_warp(image)

    name = detect_name(crop_name_area(warped))
    class_name = detect_class(crop_class_area(warped))
    section = detect_section(crop_section_area(warped))

    roll_number = detect_roll_number(
        crop_roll_area(warped),
        num_digits=2
    )

    stream = detect_stream(crop_stream_area(warped))
    set_number = detect_set(crop_set_area(warped))

    subject_code = detect_subject_code(
        crop_subject_code_area(warped)
    )

    admission_number = detect_admission_number(
        crop_admission_area(warped)
    )

    answer_area = crop_answer_area(warped)


    answers = detect_all_answers(answer_area)

    return {
        "name": name,
        "class_name": class_name,
        "section": section,
        "roll_number": roll_number,
        "stream": stream,
        "set_number": set_number,
        "subject_code": subject_code,
        "admission_number": admission_number,
        "answers": answers
    }