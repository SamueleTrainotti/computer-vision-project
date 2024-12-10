# from calibration import Calibration
import calibration
import sys

def main():
    camera_calibration = Calibration()
    camera_calibration.calibrate()


    print("Hello World!")

if __name__ == "__main__":
    sys.path += [r"D:\Download\CV_proj\.venv"]
    main()