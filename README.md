# Computer Vision Project

This repository contains the **Computer Vision project** focused on the calibration and visualization of 3D human models using self-centric and stereo camera systems. The project leverages Python and Blender for generating accurate 2D and 3D data representations.


## Project Overview
The project is divided into two distinct tasks:

### Task 1: Single Camera Calibration
- **Objective**: Position a camera at eye level in Blender to capture the entire body of the 3D human model.
- **Steps**:
  1. Configure a self-centric camera looking downward to frame the full body.
  2. Calibrate the camera to extract **intrinsic** and **extrinsic parameters**.
  3. Render the camera view and plot the skeletal bones on the 2D image plane.
- **Outcome**: Visualization of 2D skeletal structures aligned with the camera's perspective.

### Task 2: Stereo Camera Calibration
- **Objective**: Position a second camera near the first one to enable stereo calibration.
- **Steps**:
  1. Set up the stereo camera system in Blender.
  2. Perform stereo calibration using chessboard patterns from the `Stereo_images` folder.
  3. Extract 3D coordinates of the skeletal bones and joints through triangulation of the 2D coordinates obtained from both cameras.
- **Outcome**: Reconstruction of 3D skeletal structures from the stereo setup.

## Technologies Used
This project utilizes the following technologies and libraries:
- **Python**
- **OpenCV**
- **Blender**
- **NumPy**

## Installation
To get started with the project, follow these steps:

1. Clone the repository:
   ```bash
   git clone https://github.com/SamueleTrainotti/computer-vision-project.git
   cd computer-vision-project
   ```

2. Install the required dependencies:
   ```bash
   pip install -r requirements.txt
   ```

3. Ensure that Blender is installed and configured to execute Python scripts.

## File Organization
This repository contains the following key files and directories:

- `all.py`: A comprehensive script that includes all functionalities, covering both single-camera and stereo-camera calibration tasks. This script is also embedded in the Blender `.blend` files for seamless execution.
- `main.py`: A script specifically designed for single-camera calibration (Task 1). The logic is divided into smaller modules for ease of understanding and reusability.
- `/Blender_projects`: Contains Blender files with embedded scripts for single and stereo camera calibration. The Python script within these files is identical to the all.py script.
- `/Stereo_images`: A folder containing calibration images for the stereo setup. These images feature a chessboard pattern used to compute the intrinsic and extrinsic parameters of the two cameras.