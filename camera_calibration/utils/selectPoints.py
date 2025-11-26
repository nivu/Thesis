import cv2 as cv
from dataclasses import dataclass
import numpy as np
from tkinter import Tk, OptionMenu, StringVar, Button


@dataclass
class Position_3d:
    label: str
    x: float
    y: float
    z: float


class PointSelector:
    def __init__(self, frame, sizeZoomArea=100, zoom_factor=30):
        self.points_3d = self.read_config()

        self.frame = frame
        self.height, self.width, self.channels = frame.shape
        self.sizeZoomArea = sizeZoomArea
        self.zoom_factor = zoom_factor

        self.size_zoomedArea = sizeZoomArea * zoom_factor

        self.cursor_pos = np.array([0, 0])
        self.selected_zoomArea_position = np.array([0, 0])
        self.selected_points = []

        self.selected_point_3d = []

    def on_mouse_click(self, event, x, y, flags, param):
        if event == cv.EVENT_MOUSEMOVE:
            self.cursor_pos = np.array([x, y])

        if event == cv.EVENT_LBUTTONDOWN:
            if x > self.width and y < self.size_zoomedArea:
                x = (
                    (x - self.width) / self.zoom_factor
                    + self.selected_zoomArea_position[0]
                    - self.sizeZoomArea / 2
                )
                y = (
                    y / self.zoom_factor
                    + self.selected_zoomArea_position[1]
                    - self.sizeZoomArea / 2
                )

                is_selected = self.select_label()
                if is_selected:
                    self.points_3d.remove(self.selected_point_3d[-1])
                    self.selected_points.append(np.array([x, y]))

                    print("selected points")
                    for selected_point_3 in self.selected_point_3d:
                        print(selected_point_3.label)

                    print("avaliable points")
                    for points_3d in self.points_3d:
                        print(points_3d.label)

            elif x < self.width and y < self.height:
                self.selected_zoomArea_position = np.array([x, y])

        if event == cv.EVENT_RBUTTONDOWN:
            if len(self.selected_points) > 0:
                self.points_3d.append(self.selected_point_3d[-1])

                self.selected_points.pop()
                self.selected_point_3d.pop()

                print("Point deleted")

    def getZoomview(self):
        frame_copy = self.frame.copy()

        topLeftCornerZoomArea = (
            int(
                max(
                    self.sizeZoomArea / 2,
                    self.selected_zoomArea_position[0] - self.sizeZoomArea / 2,
                )
            ),
            int(
                max(
                    self.sizeZoomArea / 2,
                    self.selected_zoomArea_position[1] - self.sizeZoomArea / 2,
                )
            ),
        )

        bottomRightCornerZoomArea = (
            int(
                max(
                    topLeftCornerZoomArea[0] + self.sizeZoomArea,
                    self.selected_zoomArea_position[0] + self.sizeZoomArea / 2,
                )
            ),
            int(
                max(
                    topLeftCornerZoomArea[1] + self.sizeZoomArea,
                    self.selected_zoomArea_position[1] + self.sizeZoomArea / 2,
                )
            ),
        )

        zoomed_cropped = frame_copy[
            topLeftCornerZoomArea[1] : bottomRightCornerZoomArea[1],
            topLeftCornerZoomArea[0] : bottomRightCornerZoomArea[0],
        ]
        zoomed_region = cv.resize(
            zoomed_cropped,
            None,
            fx=self.zoom_factor,
            fy=self.zoom_factor,
            interpolation=cv.INTER_LINEAR,
        )

        return zoomed_region

    def select_label(
        self,
    ):
        master = Tk()
        master.title("Label Selector")

        variable = StringVar(master)
        labels = []
        for points_3d in self.points_3d:
            labels.append(points_3d.label)

        variable.set(labels[0])  # default value

        w = OptionMenu(master, variable, *labels)
        w.pack()

        def ok():
            selected_label = variable.get()
            for points_3d in self.points_3d:
                if points_3d.label == selected_label:
                    self.selected_point_3d.append(points_3d)
                    break

            self.is_selected = True
            master.destroy()

        def cancel():
            print("Selection canceled.")
            self.is_selected = False
            master.destroy()

        ok_button = Button(master, text="OK", command=ok)
        ok_button.pack(side="right", padx=20, pady=20)

        cancel_button = Button(master, text="Cancel", command=cancel)
        cancel_button.pack(side="left", padx=20, pady=20)

        master.mainloop()
        return self.is_selected

    def selectPoints(
        self,
    ):
        cv.namedWindow("select_points", cv.WINDOW_NORMAL)
        cv.setMouseCallback("select_points", self.on_mouse_click)

        sizeMainWindow = (
            self.height + self.sizeZoomArea * self.zoom_factor,
            self.width + self.sizeZoomArea * self.zoom_factor,
            self.channels,
        )
        main_window = np.zeros(sizeMainWindow, dtype=np.uint8)

        while True:
            frame_copy = self.frame.copy()
            topLeftCornerZoomArea = (self.cursor_pos[0] - 50, self.cursor_pos[1] - 50)
            bottomRightCornerZoomArea = (
                self.cursor_pos[0] + 50,
                self.cursor_pos[1] + 50,
            )

            cv.rectangle(
                frame_copy,
                topLeftCornerZoomArea,
                bottomRightCornerZoomArea,
                (0, 255, 0),
                2,
            )
            for i in self.selected_points:
                cv.circle(frame_copy, (int(i[0]), int(i[1])), 5, (0, 255, 0), -1)

            zoomed_region = self.getZoomview()

            main_window[0 : self.height, 0 : self.width] = frame_copy.copy()

            main_window[
                0 : self.sizeZoomArea * self.zoom_factor,
                self.width : self.width + self.sizeZoomArea * self.zoom_factor,
            ] = zoomed_region.copy()

            if (
                self.cursor_pos[0] > self.width
                and self.cursor_pos[1] < self.sizeZoomArea * self.zoom_factor
            ):
                cv.circle(
                    main_window,
                    (int(self.cursor_pos[0]), int(self.cursor_pos[1])),
                    3,
                    (0, 255, 0),
                    -1,
                )

            cv.imshow("select_points", main_window)
            key = cv.waitKey(1) & 0xFF

            if key == 27:  # 27 is the ASCII value for the "Esc" key
                print("Esc key pressed. Exiting.")
                cv.destroyAllWindows()
                return self.selected_point_3d, self.selected_points
            if len(self.points_3d) == 0:
                print("All points selected.")
                cv.destroyAllWindows()
                return self.selected_point_3d, self.selected_points

    def read_config(
        self,
    ):
        with open("Beispieldaten/Max_Pla.txt", "r") as file:
            lines = file.readlines()

        points_3d = []

        # Loop through each line and extract values
        for line in lines:
            # Split the line into name, x, y, and z using commas as separators
            data = line.strip().split(",")

            # Check if the line contains at least four values
            if len(data) >= 4:
                try:
                    # Extract values and append to respective lists
                    position_3d = Position_3d(
                        data[0], float(data[1]), float(data[2]), float(data[3])
                    )
                    points_3d.append(position_3d)
                except ValueError:
                    print("Invalid data found in line: %s" % line)

        return points_3d


if __name__ == "__main__":
    calibration_image = cv.imread("./Beispieldaten/example-street.png")

    pointSelector = PointSelector(calibration_image)
    selected_label, selected_points = pointSelector.selectPoints()

    print("Points selected")
