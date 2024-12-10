import os
import pkgutil
import subprocess
import sys
import bpy
from mathutils import Vector
from mathutils import Matrix
from enum import Enum

####################################################
########### INSTALL NEEDED EXTRA PACKAGES ##########
####################################################
# Get the path to the Python interpreter used by Blender
python_path = sys.executable

# Determine the script directory
script_directory = os.path.dirname(bpy.data.filepath)

# Create a folder for packages
packages_directory = os.path.join(script_directory, "packages")
os.makedirs(packages_directory, exist_ok=True)

# Print the path
print("Blender's Python path:", python_path)

# Ensure pip is installed
subprocess.run([python_path, "-m", "ensurepip"])

# Check if the required packages are installed
required_packages = ["matplotlib"]

installed_packages = {pkg.name for pkg in pkgutil.iter_modules()}

for package in required_packages:
    if package not in installed_packages:
        # Install the required package in the created folder
        subprocess.run([python_path, "-m", "pip", "install", "--target=" + packages_directory, package])
    else:
        print(f"{package} is already installed")

# Add the folder to the Python import path
sys.path.append(packages_directory)
####################################################

import cv2
import matplotlib.pyplot as plt
import numpy as np

##### Define the 2D points #####
points2dZ11 = [(346, 1053), (991, 1053), (1636, 1053), (1894, 458), (1803, 61), (1391, 61), (527, 61), (116, 61),
               (25, 458), (1515, 436), (405, 436), (1838, 262), (80, 262)]
points2dZ9 = [(70, 971), (504, 971), (939, 971), (1372, 971), (1806, 971), (1667, 517), (1555, 239), (363, 239),
              (252, 517), (1321, 517), (598, 517)]
points2dZ7 = [(341, 879), (643, 879), (945, 879), (1247, 879), (1548, 879), (1484, 556), (1420, 337), (499, 337),
              (435, 556), (1227, 556), (691, 556)]
points2dZ5 = [(486, 829), (718, 829), (949, 829), (1180, 829), (1410, 829), (1342, 583), (577, 583), (1173, 579),
              (747, 580)]
points2dZ3 = [(576, 798), (764, 798), (951, 798), (1138, 798), (1324, 798), (1305, 595), (1276, 441), (643, 441),
              (614, 595), (1136, 595), (783, 595)]

##### Define the 3D points #####
centerCubes3d = [(-8, -4, 0), (-4, -4, 0), (0, -4, 0), (4, -4, 0), (8, -4, 0), (8, 0, 0), (8, 4, 0), (-8, 4, 0),
                 (-8, 0, 0), (4, 0, 0), (-4, 0, 0)]
points3dZ3, points3dZ5, points3dZ7, points3dZ9, points3dZ11 = [], [], [], [], []
points3dZ11 += [(x + 0.2, y + 0.2, z + 10.8) for x, y, z in centerCubes3d[1:4]]
points3dZ9 += [(x - 0.2, y - 0.2, z + 9.2) for x, y, z in centerCubes3d[0:5]]
points3dZ7 += [(x - 0.2, y - 0.2, z + 7.2) for x, y, z in centerCubes3d[0:5]]
points3dZ5 += [(x - 0.2, y - 0.2, z + 5.2) for x, y, z in centerCubes3d[0:5]]
points3dZ3 += [(x - 0.2, y - 0.2, z + 3.2) for x, y, z in centerCubes3d[0:5]]

points3dZ11 += [(centerCubes3d[5][0] - 0.2, centerCubes3d[5][1] + 0.2, centerCubes3d[5][2] + 10.8)]
points3dZ11 += [(centerCubes3d[6][0] + 0.2, centerCubes3d[6][1] + 0.2, centerCubes3d[6][2] + 11.2)]
points3dZ9 += [(x + 0.2, y + 0.2, z + 9.2) for x, y, z in centerCubes3d[5:7]]
points3dZ7 += [(x + 0.2, y + 0.2, z + 7.2) for x, y, z in centerCubes3d[5:7]]
points3dZ3 += [(x + 0.2, y + 0.2, z + 3.2) for x, y, z in centerCubes3d[5:7]]
points3dZ5 += [(centerCubes3d[5][0] - 0.2, centerCubes3d[5][1] + 0.2, centerCubes3d[5][2] + 4.8)]

points3dZ11 += [(4.2, 4.2, 11.2), (-4.2, 4.2, 11.2)]

points3dZ11 += [(centerCubes3d[7][0] - 0.2, centerCubes3d[7][1] + 0.2, centerCubes3d[7][2] + 11.2)]
points3dZ11 += [(centerCubes3d[8][0] + 0.2, centerCubes3d[8][1] + 0.2, centerCubes3d[8][2] + 10.8)]
points3dZ9 += [(x - 0.2, y + 0.2, z + 9.2) for x, y, z in centerCubes3d[7:9]]
points3dZ7 += [(x - 0.2, y + 0.2, z + 7.2) for x, y, z in centerCubes3d[7:9]]
points3dZ3 += [(x - 0.2, y + 0.2, z + 3.2) for x, y, z in centerCubes3d[7:9]]
points3dZ5 += [(centerCubes3d[8][0] + 0.2, centerCubes3d[8][1] + 0.2, centerCubes3d[8][2] + 4.8)]

points3dZ11 += [(centerCubes3d[9][0] + 0.2, centerCubes3d[9][1] + 0.2, centerCubes3d[9][2] + 11.2),
                (centerCubes3d[10][0] - 0.2, centerCubes3d[10][1] + 0.2, centerCubes3d[10][2] + 11.2)]
points3dZ9 += [(centerCubes3d[9][0] + 0.2, centerCubes3d[9][1] + 0.2, centerCubes3d[9][2] + 9.2),
               (centerCubes3d[10][0] - 0.2, centerCubes3d[10][1] + 0.2, centerCubes3d[10][2] + 9.2)]
points3dZ7 += [(centerCubes3d[9][0] + 0.2, centerCubes3d[9][1] + 0.2, centerCubes3d[9][2] + 7.2),
               (centerCubes3d[10][0] - 0.2, centerCubes3d[10][1] + 0.2, centerCubes3d[10][2] + 7.2)]
points3dZ5 += [(centerCubes3d[9][0] + 0.2, centerCubes3d[9][1] + 0.2, centerCubes3d[9][2] + 5.2),
               (centerCubes3d[10][0] - 0.2, centerCubes3d[10][1] + 0.2, centerCubes3d[10][2] + 5.2)]
points3dZ3 += [(centerCubes3d[9][0] + 0.2, centerCubes3d[9][1] + 0.2, centerCubes3d[9][2] + 3.2),
               (centerCubes3d[10][0] - 0.2, centerCubes3d[10][1] + 0.2, centerCubes3d[10][2] + 3.2)]

points3dZ11 += [(3.8, 0.2, 12.8), (-3.8, +0.2, 12.8)]


##### CALIBRATION CLASS #####
class Calibration:

    def __init__(self, image_path):
        self.FOCAL_LENGTH = 9  # Focal length in mm
        self.SENSOR_SIZE = (36, 24)  # Sensor size in mm
        self.RESOLUTION = (1920, 1080)  # Image resolution in pixels
        self.SCALE = 1  # Scale factor

        self.image_path = image_path
        self.points3d = points3dZ3 + points3dZ5 + points3dZ7 + points3dZ9 + points3dZ11
        self.points2d = points2dZ3 + points2dZ5 + points2dZ7 + points2dZ9 + points2dZ11

        self.distortion_coefficients = np.zeros((4, 1))  # Assuming no lens distortion
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
            [0, fy, cy],
            [0, 0, 1]])

        return K

    def calibrate(self):
        # Get the intrinsic parameters
        self.K = self.get_intrinsic_parameters()
        print("Intrinsic Matrix:", self.K)

        # Convert the points to numpy array
        pts_2d = np.array(self.points2d, dtype=np.float32)
        pts_3d = np.array(self.points3d, dtype=np.float32)
        print(len(pts_2d), len(pts_3d))
        success, self.rotation_vector, self.translation_vector = cv2.solvePnP(pts_3d, pts_2d, self.K,
                                                                              self.distortion_coefficients)

        # Convert the rotation vector to a rotation matrix
        rotation_matrix, _ = cv2.Rodrigues(self.rotation_vector)

        # Combine the rotation matrix and translation vector
        extrinsic_matrix = np.hstack((rotation_matrix, self.translation_vector))

        # Compute the projection matrix
        self.P = self.K.dot(extrinsic_matrix)
        print("Projection Matrix:", self.P)

        object_points = np.array([[1.2, 0, 13.5], [1, 1.5, 12], [3, 3, 13], [-3, 3, 10], [0, -2, 8]], dtype=np.float32)

        self.projectPoints(object_points)

        # cv2.waitKey(0)
        # cv2.destroyAllWindows()

    def projectPoints(self, object_points):
        image_points, _ = cv2.projectPoints(object_points, self.rotation_vector, self.translation_vector, self.K,
                                            self.distortion_coefficients)

        image_points_list = [tuple(point[0]) for point in image_points]
        print("Image Points:", image_points_list)

        self.show_points_on_image(self.image_path, image_points_list)
        # plot_3d_points(object_points, _3d=False)


#############################################
class ModelBones:
    def __init__(self):
        self.bones_3d = []
        body = bpy.data.objects["Armature"]

        if body:
            for bone in body.data.bones:
                tail = body.matrix_world @ bone.tail_local
                head = body.matrix_world @ bone.head_local

                self.bones_3d.append(tuple((Vector(tail), Vector(head))))
            print(len(self.bones_3d))
        else:
            print("Nessuna armatura trovata.")

    def get_bones_2d(self, matrix):
        matrix = Matrix(matrix)
        bones_2d = []
        for p1_3d, p2_3d in self.bones_3d:
            p1_2d = self.compute_2d(p1_3d, matrix)
            p2_2d = self.compute_2d(p2_3d, matrix)
            bones_2d.append(tuple((p1_2d, p2_2d)))
        return bones_2d

    def compute_2d(self, point, matrix):
        point = Vector(point)
        tmp = point.to_4d()
        p_2d = matrix @ tmp
        p_2d /= p_2d[2]
        p_2d = p_2d.to_2d().to_tuple(0)
        p_2d = tuple(int(num) for num in p_2d)
        return p_2d


##################################################################
def draw_bones(bones_2d, filename, output):
    # Try to draw all bones on final image
    print("PRINTING BONES ON IMAGE")
    img = cv2.imread(filename)
    for p1, p2 in bones_2d:
        img = cv2.line(img, p1, p2, (0, 0, 255))

    cv2.imwrite(output, img)
    cv2.imshow('output', img)

    cv2.waitKey(0)
    cv2.destroyAllWindows()


##################################################################
# https://github.com/bvnayak/stereo_calibration/blob/master/camera_calibrate.py
class CameraView(Enum):
    LEFT = 0
    RIGHT = 1

class StereoCalibration():
    def __init__(self, filepath):
        # termination criteria
        self.criteria = (cv2.TERM_CRITERIA_EPS +
                         cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)
        self.criteria_cal = (cv2.TERM_CRITERIA_EPS +
                             cv2.TERM_CRITERIA_MAX_ITER, 100, 1e-5)

        # prepare object points
        # first (i.e. 0,0) in (-4.99753, -3.37665)
        # second (i.e. in 0, 1) in (-3.74853, -3.37665)
        # each square has size 1,2494 x -1,2494
        self.objp = np.full((9 * 6, 3), [-4.99753, -3.37665, 0], np.float32)
        offsets = np.ones((9 * 6, 3), np.float32)
        for i in range(9 * 6):
            offsets[i] = [(i // 9) * 1.2494, (i % 9) * -1.2494, 0]
        # print("OFFSET")
        # print(offsets)
        self.objp = np.subtract(self.objp, offsets)

        # Arrays to store object points and image points from all the images.
        self.objpoints = []  # 3d point in real world space
        self.imgpoints_l = []  # 2d points in image plane.
        self.imgpoints_r = []  # 2d points in image plane.

        self.cal_path = filepath
        self.read_images(self.cal_path)

    def read_images(self, cal_path):
        images_right = [os.path.join(cal_path, 'Stereo_images', 'calibrate_2_left.png')]
        images_left = [os.path.join(cal_path, 'Stereo_images', 'calibrate_2_right.png')]

        for i, fname in enumerate(images_right):
            img_l = cv2.imread(images_left[i])
            img_r = cv2.imread(images_right[i])

            gray_l = cv2.cvtColor(img_l, cv2.COLOR_BGR2GRAY)
            gray_r = cv2.cvtColor(img_r, cv2.COLOR_BGR2GRAY)

            # Find the chess board corners
            ret_l, corners_l = cv2.findChessboardCorners(gray_l, (9, 6), None)
            ret_r, corners_r = cv2.findChessboardCorners(gray_r, (9, 6), None)

            # If found, add object points, image points (after refining them)
            self.objpoints.append(self.objp)

            if ret_l is True:
                rt = cv2.cornerSubPix(gray_l, corners_l, (11, 11),
                                      (-1, -1), self.criteria)
                self.imgpoints_l.append(corners_l)

                # Draw and display the corners
                ret_l = cv2.drawChessboardCorners(img_l, (9, 6),
                                                  corners_l, ret_l)

                img_l = cv2.resize(img_l, (960, 540))
                cv2.imshow(images_left[i], img_l)
                cv2.waitKey(500)

            if ret_r is True:
                rt = cv2.cornerSubPix(gray_r, corners_r, (11, 11),
                                      (-1, -1), self.criteria)
                self.imgpoints_r.append(corners_r)

                # Draw and display the corners
                ret_r = cv2.drawChessboardCorners(img_r, (9, 6),
                                                  corners_r, ret_r)

                img_r = cv2.resize(img_r, (960, 540))
                cv2.imshow(images_right[i], img_r)
                cv2.waitKey(500)
            img_shape = gray_l.shape[::-1]

        # print("3D POINTS")
        # print(self.objpoints)
        # print("2D LEFT")
        # print(self.imgpoints_l)
        # print("2D RIGHT")
        # print(self.imgpoints_r)
        rt, self.M1, self.d1, self.r1, self.t1 = cv2.calibrateCamera(
            self.objpoints, self.imgpoints_l, img_shape, None, None)
        rt, self.M2, self.d2, self.r2, self.t2 = cv2.calibrateCamera(
            self.objpoints, self.imgpoints_r, img_shape, None, None)

        self.camera_model = self.stereo_calibrate(img_shape)

    def stereo_calibrate(self, dims):
        flags = 0
        flags |= cv2.CALIB_FIX_INTRINSIC
        # flags |= cv2.CALIB_FIX_PRINCIPAL_POINT
        flags |= cv2.CALIB_USE_INTRINSIC_GUESS
        flags |= cv2.CALIB_FIX_FOCAL_LENGTH
        # flags |= cv2.CALIB_FIX_ASPECT_RATIO
        flags |= cv2.CALIB_ZERO_TANGENT_DIST
        # flags |= cv2.CALIB_RATIONAL_MODEL
        # flags |= cv2.CALIB_SAME_FOCAL_LENGTH
        # flags |= cv2.CALIB_FIX_K3
        # flags |= cv2.CALIB_FIX_K4
        # flags |= cv2.CALIB_FIX_K5

        stereocalib_criteria = (cv2.TERM_CRITERIA_MAX_ITER +
                                cv2.TERM_CRITERIA_EPS, 100, 1e-5)
        ret, M1, d1, M2, d2, R, T, E, F = cv2.stereoCalibrate(
            self.objpoints, self.imgpoints_l,
            self.imgpoints_r, self.M1, self.d1, self.M2,
            self.d2, dims,
            criteria=stereocalib_criteria, flags=flags)

        # print('Intrinsic_mtx_1', M1)
        # print('dist_1', d1)
        # print('Intrinsic_mtx_2', M2)
        # print('dist_2', d2)
        # print('R', R)
        # print('T', T)
        # print('E', E)
        # print('F', F)

        # for i in range(len(self.r1)):
        #     print("--- pose[", i+1, "] ---")
        #     self.ext1, _ = cv2.Rodrigues(self.r1[i])
        #     self.ext2, _ = cv2.Rodrigues(self.r2[i])
        #     print('Ext1', self.ext1)
        #     print('Ext2', self.ext2)

        print('')

        camera_model = dict([('M1', M1), ('M2', M2), ('dist1', d1),
                             ('dist2', d2), ('rvecs1', self.r1),
                             ('rvecs2', self.r2), ('tvecs1', self.t1),
                             ('tvecs2', self.t2),('R', R), ('T', T),
                             ('E', E), ('F', F)])

        cv2.destroyAllWindows()
        return camera_model

    def reProjectionError(self, imgpoints, n = '1'):
        mean_error = 0
        mtx = self.camera_model['M'+n]
        dist = self.camera_model['dist'+n]
        rvecs = self.camera_model['rvecs'+n]
        tvecs = self.camera_model['tvecs'+n]

        for i in range(len(self.objpoints)):
            imgpoints2, _ = cv2.projectPoints(self.objpoints[i], rvecs[i], tvecs[i], mtx, dist)
            error = cv2.norm(imgpoints[i], imgpoints2, cv2.NORM_L2)/len(imgpoints2)
            mean_error += error

        print( "total error: {}".format(mean_error/len(self.objpoints)) )

    def getProjectionMatrix(self, side = CameraView.LEFT):
        if(side == CameraView.LEFT):
            mtx = self.camera_model['M1']
            rvecs = self.camera_model['rvecs1']
            tvecs = self.camera_model['tvecs1']
        else:
            mtx = self.camera_model['M2']
            rvecs = self.camera_model['rvecs2']
            tvecs = self.camera_model['tvecs2']

        R = cv2.Rodrigues(rvecs[0])[0]
        t = tvecs[0]
        Rt = np.concatenate([R,t], axis=-1) # [R|t]
        P = np.matmul(mtx,Rt) # A[R|t]
        print(P)
        return P

    def triangulate3D(self, leftPoints, rightPoints):
        m1 = self.getProjectionMatrix(CameraView.LEFT)
        m2 = self.getProjectionMatrix(CameraView.RIGHT)
        leftPoints = np.transpose(leftPoints)
        rightPoints = np.transpose(rightPoints)
        # print(f"M1: type {type(m1)}\n", m1)
        # print(f"M2: type {type(m2)}\n", m2)
        # print(f"Left: type {type(leftPoints)}\n", leftPoints)
        # print(leftPoints.shape)
        # print(f"Right: type {type(rightPoints)}\n", rightPoints)
        # print(rightPoints.shape)
        return cv2.triangulatePoints(m1, m2, leftPoints, rightPoints)


########## ALL PUT TOGHETER ##########
def main():
    base_dir = r"D:\Download\CV_proj\.venv"  ### CHANGE THIS !!! ###
    '''baseline_path = os.path.join(base_dir, "baseline_1.png")
    output_path = os.path.join(base_dir, "output.png")

    camera_calibration = Calibration(baseline_path)
    camera_calibration.calibrate()

    model_bones = ModelBones()
    bones_2d = model_bones.get_bones_2d(camera_calibration.P)
    draw_bones(bones_2d, baseline_path, output_path)'''

    cal_data = StereoCalibration(base_dir)
    cal_data.reProjectionError(cal_data.imgpoints_l, '1')
    cal_data.reProjectionError(cal_data.imgpoints_r, '2')
    # cal_data.getProjectionMatrix()
    # cal_data.getProjectionMatrix(CameraView.RIGHT)
    # TEST POINTS -> pollice1, mento, board1, board2
    # LEFT: (460, 665) - (1035, 401) - (1188, 686) - (715, 1062)
    # RIGHT: (422, 665) - (805, 400) - (1174, 686) - (699, 1061)
    leftPoints = np.asarray([[460.0, 665.0], [1035.0, 401.0], [1188.0, 686.0], [715.0, 1062.0]])
    rightPoints = np.asarray([[422.0, 665.0], [805.0, 400.0], [1174.0, 686.0], [699.0, 1061.0]])
    points_h = cal_data.triangulate3D(leftPoints, rightPoints)
    print("homogeneous points\n", points_h)
    points_h /= points_h[3]
    points_3D = points_h[:3, :].T
    print("3D\n", points_3D)


if __name__ == "__main__":
    main()
