import numpy as np
from bokeh.plotting import figure

def wiggle(data, tt=None, xx=None, color='black', stretch_factor=0.15, verbose=False):
    
    # Input check
    data, tt, xx, trace_spacing = wiggle_input_check(data, tt, xx, stretch_factor, verbose)    
    number_of_traces = data.shape[1]

    plot = figure(
        x_range=(xx[0] - trace_spacing, xx[-1] + trace_spacing),
        y_range=(tt[0], tt[-1]),
        x_axis_location='above',
    )
    
    for trace_index in range(number_of_traces):
        trace = data[:, trace_index]
        offset = xx[trace_index]
        plot.line(x=trace + offset, y=tt, color=color)

    return plot


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
