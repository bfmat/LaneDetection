# LaneDetection
A lane detection system for autonomous driving in Keras, loosely based on https://github.com/brendon-ai/SelfDrivingNetwork but not end-to-end.

## Usage Notes
In order to run the executable scripts, which import other modules within the project, you may have to add the parent directory of the project to your `PYTHONPATH`. You must execute the scripts with `python -m LaneDetection.<subpackage>.<script>`. Running them directly with `./<script>` will result in an import error.

## Compatibility
This project is intended to be used with Python 2.7. Most scripts should work with 3.x, but this is not guaranteed.
