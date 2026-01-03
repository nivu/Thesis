import numpy as np

def generate_bins(bins):
        """Generate angle bins for orientation estimation"""
        angle_bins = np.zeros(bins)
        interval = 2 * np.pi / bins
        for i in range(1, bins):
            angle_bins[i] = i * interval
        angle_bins += interval / 2  # center of bins
        return angle_bins

def calc_theta_ray(image_width, box_2d, fx):
        """Calculate global angle of object"""
        fovx = 2 * np.arctan(image_width / (2 * fx))
        center = (box_2d[0] + box_2d[2]) / 2
        dx = center - (image_width / 2)

        mult = 1 if dx >= 0 else -1
        dx = abs(dx)
        angle = np.arctan((2 * dx * np.tan(fovx / 2)) / image_width)
        angle = angle * mult

        return angle

def get_bin(angle, nb_bins, overlap):

    interval = 2 * np.pi / nb_bins

    bin_ranges = []
    for i in range(0, nb_bins):
        bin_ranges.append(((i * interval - overlap) % (2 * np.pi),
                            (i * interval + interval + overlap) % (2 * np.pi)))

    bin_idxs = []

    def is_between(min, max, angle):
        max = (max - min) if (max - min) > 0 else (max - min) + 2*np.pi
        angle = (angle - min) if (angle - min) > 0 else (angle - min) + 2*np.pi
        return angle < max

    for bin_idx, bin_range in enumerate(bin_ranges):
        if is_between(bin_range[0], bin_range[1], angle):
            bin_idxs.append(bin_idx)

    return bin_idxs

def get_angle_from_bins(confidence, orientation, nb_bins):

    angle_bins = generate_bins(nb_bins)

    argmax = np.argmax(confidence)
    orient = orientation[argmax, :]
    cos, sin = orient[0], orient[1]
    alpha = np.arctan2(sin, cos)
    alpha += angle_bins[argmax]
    alpha -= np.pi

    return alpha