import cv2


def run_template_match(image, template, threshold=0.9):
    res = cv2.matchTemplate(image=image, templ=template, method=cv2.TM_CCOEFF_NORMED)
    min_max = cv2.minMaxLoc(res)
    max_value = min_max[1]
    max_location = min_max[3]
    if max_value > threshold:
        found = True
    else:
        found = False

    return found, max_location, max_value
