import matplotlib.pyplot as plt
import numpy as np


# data.shape[0] -> number of rows -> number of samples
# data.shape[1] -> number of columns -> number of traces

# each column is a whole trace
# within a column, each line is a sample
# each line is a collection of samples collected at the same time


def wiggle(data, tt=None, xx=None, color='k', stretch_factor=0.15, verbose=False):
    """Wiggle plot of a seismic data section

    Syntax examples:
        wiggle(data)
        wiggle(data, tt)
        wiggle(data, tt, xx)
        wiggle(data, tt, xx, color)
        fi = wiggle(data, tt, xx, color, sf, verbose)

    Use the column major order for array as in Fortran for optimal performance
    """
    # Input check
    data, tt, xx, trace_spacing = wiggle_input_check(data, tt, xx, stretch_factor, verbose)

    # Plot data using matplotlib.pyplot

    number_of_traces = data.shape[1]

    # Get the current Axes
    ax = plt.gca()

    for trace_index in range(number_of_traces):
        trace = data[:, trace_index]
        offset = xx[trace_index]
        ax.plot(trace + offset, tt, color=color)

    ax.set_xlim(xx[0] - trace_spacing, xx[-1] + trace_spacing)
    ax.set_ylim(tt[0], tt[-1])
    ax.invert_yaxis()


def wiggle_input_check(data, tt, xx, stretch_factor, verbose):
    # Input check for verbose
    if not isinstance(verbose, bool):
        raise TypeError("verbose must be a bool")

    # Input check for data
    if type(data).__module__ != np.__name__:
        raise TypeError("data must be a numpy array")
    if len(data.shape) != 2:
        raise ValueError("data must be a 2D array")

    # Input check for tt
    if tt is None:
        tt = np.arange(data.shape[0])
        # if verbose:
        #     print("tt is automatically generated.")
        #     print(tt)
    else:
        if type(tt).__module__ != np.__name__:
            raise TypeError("tt must be a numpy array")
        if len(tt.shape) != 1:
            raise ValueError("tt must be a 1D array")
        if tt.shape[0] != data.shape[0]:
            raise ValueError("tt's size must be the equal to to the number of rows in data, "
                             "that is, equal to the number of samples")

    # Input check for xx
    if xx is None:
        xx = np.arange(data.shape[1])
        # if verbose:
        #     print("xx is automatically generated")
        #     print(xx)
    else:
        if type(xx).__module__ != np.__name__:
            raise TypeError("xx must be a numpy array")
        if len(xx.shape) != 1:
            raise ValueError("xx must be a 1D array")
        if xx.shape[0] != data.shape[1]:
            raise ValueError("xx's size must be equal to the number of columns in data, "
                             "that is, equal to the number of traces")

    # Input check for stretch factor (sf)
    if not isinstance(stretch_factor, (int, float)):
        raise TypeError("stretch_factor must be a number")

    # Compute trace horizontal spacing
    trace_spacing = np.min(np.diff(xx))

    # Rescale data by trace_spacing and stretch_factor
    data_max_std = np.max(np.std(data, axis=0))
    data = data / data_max_std * trace_spacing * stretch_factor

    return data, tt, xx, trace_spacing


def insert_zeros(trace, tt=None):
    """Insert zero locations in data trace and tt vector based on linear fit"""

    if tt is None:
        tt = np.arange(len(trace))

    # Find zeros
    zc_index = np.where(np.diff(np.signbit(trace)))[0]

    x1 = tt[zc_index]
    x2 = tt[zc_index + 1]

    y1 = trace[zc_index]
    y2 = trace[zc_index + 1]

    a = (y2 - y1) / (x2 - x1)
    tt_zero = x1 - y1 / a

    # split tt and trace
    tt_split = np.split(tt, zc_index + 1)
    trace_split = np.split(trace, zc_index + 1)

    tt_zi = tt_split[0]
    trace_zi = trace_split[0]

    # insert zeros in tt and trace
    for i in range(len(tt_zero)):
        tt_zi = np.hstack(
            (tt_zi, np.array([tt_zero[i]]), tt_split[i + 1]))
        trace_zi = np.hstack(
            (trace_zi, np.zeros(1), trace_split[i + 1]))

    return trace_zi, tt_zi
