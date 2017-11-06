import numpy


# Function to calculate a line of best fit for a set of points (Y, X format is assumed)
def line_of_best_fit(points):
    # Get an array of X positions from the points
    x = numpy.array([point[1] for point in points])

    # Get a list of Y positions, with a bias term of 1 at the beginning of each row
    y = numpy.array([[1, point[0]] for point in points])

    # Use the normal equation to find the line of best fit
    y_transpose = y.transpose()
    line_parameters = numpy.linalg.pinv(
        y_transpose.dot(y)).dot(y_transpose).dot(x)

    return line_parameters
