import cv2
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

from points2d import *
from points3d import *

class Calibration:

    def __init__(self):
        self.FOCAL_LENGTH = 9  # Focal length in mm
        self.SENSOR_SIZE = (36, 24)  # Sensor size in mm
        self.RESOLUTION = (1920, 1080)  # Image resolution in pixels
        self.SCALE = 1  # Scale factor

        self.image_path = r'D:\Download\CV_proj\.venv\baseline.png'  # Sostituisci con il percorso della tua immagine
        self.points3d = points3dZ3 + points3dZ5 + points3dZ7 + points3dZ9 + points3dZ11
        self.points2d = points2dZ3 + points2dZ5 + points2dZ7 + points2dZ9 + points2dZ11

        self.distortion_coefficients = np.zeros((4, 1)) # Assuming no lens distortion
        self.K = None
        self.rotation_vector = None
        self.translation_vector = None
        self.P = None

    def show_points_on_image(self, image_path, points):
        # Load the image
        image = cv2.imread(image_path)

        # Check if image is loaded fine
        if image is None:
            print("Error: File not found")
            return

        # Draw points and numbers on the image
        for i, point in enumerate(points):
            x, y = int(point[0]), int(point[1])
            # Draw a circle on the image using the point coordinates
            cv2.circle(image, (x, y), radius=5, color=(0, 0, 255), thickness=-1)

            # Define the text position and font
            text_position = (x + 10, y - 10)
            font = cv2.FONT_HERSHEY_SIMPLEX

            # Put the text (point number) on the image
            cv2.putText(image, str(i + 1), text_position, font, fontScale=0.5, color=(255, 255, 255), thickness=2)

        # Resize the image
        resized_image = cv2.resize(image, (1280, 720), interpolation=cv2.INTER_AREA)

        # Show the image with points and numbers
        cv2.imshow('Image with Points', resized_image)
        cv2.waitKey(0)
        cv2.destroyAllWindows()

    def plot_3d_points(self, points, _3d=True):

        # Estract the x, y and z coordinates from the points
        x_coords = [point[0] for point in points]
        y_coords = [point[1] for point in points]
        z_coords = [point[2] for point in points]

        # Create a 3D plot if _3d is True
        if _3d:
            # Create a 3D plot
            fig = plt.figure()
            ax = fig.add_subplot(111, projection='3d')

            # Plot the points
            ax.scatter(x_coords, y_coords, z_coords, c='r', marker='o')
            # Mark the points with their index in the array
            for i, (x, y, z) in enumerate(points):
                ax.text(x, y, z, '%d' % (i + 1), size=12, zorder=1, color='k')

            ax.set_xlabel('X Axis')
            ax.set_ylabel('Y Axis')
            ax.set_zlabel('Z Axis')
        else:
            # Create a 2D plot if _3d is False
            plt.scatter(x_coords, y_coords, c='r', marker='o')

            # Mark the points with their index in the array
            for i, (x, y) in enumerate(zip(x_coords, y_coords)):
                plt.text(x, y, '%d' % (i + 1), fontsize=12, ha='right', color='k')

            plt.xlabel('X Axis')
            plt.ylabel('Y Axis')

        # Show the plot
        plt.grid(True)
        plt.show()


    def get_intrinsic_parameters(self):
        # Parameters of intrinsic matrix K
        fx = self.FOCAL_LENGTH * self.RESOLUTION[0] * self.SCALE / self.SENSOR_SIZE[0]
        fy = self.FOCAL_LENGTH * self.RESOLUTION[1] * self.SCALE / self.SENSOR_SIZE[1]
        cx = self.RESOLUTION[0] * self.SCALE / 2
        cy = self.RESOLUTION[1] * self.SCALE / 2

        K = np.array([
            [fx, 0, cx],
            [0 ,fy, cy],
            [0 , 0, 1 ]])

        return K

    def calibrate(self):
        # Get the intrinsic parameters
        self.K = self.get_intrinsic_parameters()
        print("Intrinsic Matrix:", self.K)

        # Convert the points to numpy array
        pts_2d = np.array(self.points2d, dtype=np.float32)
        pts_3d = np.array(self.points3d, dtype=np.float32)
        print(len(pts_2d), len(pts_3d))
        success, self.rotation_vector, self.translation_vector = cv2.solvePnP(pts_3d, pts_2d, self.K, self.distortion_coefficients)

        # Convert the rotation vector to a rotation matrix
        rotation_matrix, _ = cv2.Rodrigues(self.rotation_vector)

        # Combine the rotation matrix and translation vector
        extrinsic_matrix = np.hstack((rotation_matrix, self.translation_vector))

        # Compute the projection matrix
        self.P = self.K.dot(extrinsic_matrix)
        print("Projection Matrix:", self.P)

        object_points = np.array([[1.2, 0, 13.5],[1, 1.5, 12], [3, 3, 13], [-3, 3, 10], [0, -2, 8]], dtype=np.float32)

        self.projectPoints(object_points)

        # cv2.waitKey(0)
        # cv2.destroyAllWindows()

    def projectPoints(self, object_points):
        image_points, _ = cv2.projectPoints(object_points, self.rotation_vector, self.translation_vector, self.K, self.distortion_coefficients)

        image_points_list = [tuple(point[0]) for point in image_points]
        print("Image Points:", image_points_list)

        self.show_points_on_image(self.image_path, image_points_list)
        #plot_3d_points(object_points, _3d=False)
