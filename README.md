# LaneDetection
A lane detection system for autonomous driving in Keras that makes use of sliding windows and proportional steering control. A visualizer is included to help with debugging and testing models.

## Usage Notes
In order to run the executable scripts, which import other modules within the project, you may have to add the parent directory of the project to your `PYTHONPATH`. You must execute the scripts with `python -m LaneDetection.<subpackage>.<script>`. Running them directly with `./<script>` will result in an import error.

## Compatibility
This project is intended to be used with Python 2.7. Many scripts will not work with Python 3.x.
