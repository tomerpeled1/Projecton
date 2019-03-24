"""
Implementation of function for tracking fruits.
"""

# Maximal distance a fruit can pass between two frames.
MOVEMENT_RADIUS = 400

# Factor to resize window for inner histogram.
RESIZE_WINDOW_FACTOR = 0.2


def center(rectangle):
    """
    calculates a center of a
    :param rectangle: the rectangle represented by top left and bottom right corner.
    :return: the center in order (width, height)
    """
    # returns in format (width, height)
    x = (rectangle[0][0] + rectangle[1][0]) / 2.0
    y = (rectangle[0][1] + rectangle[1][1]) / 2.0
    return x, y


def dis(a, b):
    """
    Euclidean distance between to points.
    :param a: First point.
    :param b: Second point.
    :return: distance between a and b.
    """
    return pow(a[0] - b[0], 2) + pow(a[1] - b[1], 2)


def update_falling(fruit):
    """
    Updates the fruit status on whether it is falling by comparison of two centers.
    :param fruit: The fruit which we want to update.
    """
    if (len(fruit.centers) < 3):
        return
    if fruit.centers[0][0][1] <= fruit.centers[1][0][1] or fruit.centers[0][0][1] <= fruit.centers[2][0][1]:
        fruit.is_falling = True


def track_object(detection_results, fruit):
    """
    Responsible for tracking the fruits. It suppose to be smart - which means to set some thresholds for
    whether we found the fruit.
    :param detection_results: The data from DETECTION. A DetectionResults object.
    :param fruit: The fruit object we know from previous frames.
    :return: True if found the fruit in the next frame, false otherwise.
    """
    if len(detection_results.centers) > 0:
        # Turns the fruit into a rectangle represented by top left and bottom right corners.
        x, y, w, h = fruit.track_window
        r = [(x, y), (x + w, y + h)]
        # Calculates the center of the rectangle.
        r_cent = center(r)
        n = len(detection_results.centers)
        min_cen = detection_results.centers[0]
        min_dis = dis(r_cent, min_cen)
        index = 0
        # Finds the contour with minimal distance to tracker results.
        for i in range(1, n):
            if dis(detection_results.centers[i], r_cent) < min_dis:
                min_cen = detection_results.centers[i]
                min_dis = dis(r_cent, min_cen)
                index = i
        # Threshold - if the fruit found is too far from original fruit.
        if min_dis > MOVEMENT_RADIUS:
            # print("min dis: " + str(min_dis))
            return False
        else:
            # This means we tracked the fruit we were looking for.
            # removes the contour so we won't look for it again.
            detection_results.conts.pop(index)
            old_track_window = rect2window(detection_results.rects.pop(index))
            detection_results.centers.pop(index)
            new_track_window = resize_track_window(old_track_window)
            # Updates the fruit track window with new frame. TODO - Try to take the meanshift window instead.
            fruit.track_window = new_track_window
            # At the end, we add another center.
            fruit.centers.append((min_cen, fruit.correlation))
            # Update the fruit's falling status.
            update_falling(fruit)
            return True


def resize_track_window(track_window):
    """
    For tracking with inner histogram.
    :param track_window: original track window
    :return: resized window
    """
    x, y, w, h = track_window
    factor = RESIZE_WINDOW_FACTOR
    inner_window = (int(x + factor * w), int(factor * h + y), int((1 - 2 * factor) * w), int((1 - 2 * factor) * h))
    return inner_window


def rect2window(rectangle):
    """
    Converts a rectangle object represented as an array with top left and bottom right object to track window.
    :param rectangle: The rectangle to convert.
    :return: The matching track window represented by (x, y, w, h)
    """
    x = rectangle[0][0]
    y = rectangle[0][1]
    w = rectangle[1][0] - x
    h = rectangle[1][1] - y
    return x, y, w, h
