from argparse import ArgumentParser
import cv2 as cv
import numpy as np

from utils.calibration import calibrate, get_lookup_table
from utils.selectPoints import PointSelector

if __name__ == "__main__":
    parser = ArgumentParser(
        description="Calibrate the image and generate a lookup table"
    )
    parser.add_argument(
        "-f",
        "--file",
        help="calibration image file",
        default="./Beispieldaten/example-street.png",
    )
    args = parser.parse_args()

    image = cv.imread("./Beispieldaten/example-street.png")

    # Select the points for calibration
    pointSelector = PointSelector(image)
    selected_label, selected_points = pointSelector.selectPoints()

    # Convert to numpy arrays
    world_points = np.array([[point.x, point.y, point.z] for point in selected_label])
    image_points = np.array([point[1], point[0]] for point in selected_points)

    # Calibrate the image and generate a lookup table
    # Use it with lookup_table[left, top]
    H, axes = calibrate(image_points, world_points)
    lookup_table = get_lookup_table(
        width=image.shape[1], height=image.shape[0], H=H, axes=axes
    )

    np.save("calibration/calibration-lookup-table", lookup_table)
