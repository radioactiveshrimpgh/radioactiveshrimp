def diff(t, x):
    """
    Calculate discrete derivatives for timeseries data.
    
    A discrete derivative is the quotient of the change in value of 
    one variable, x, and the change in value of another, t.
    
    Formula: v(t) = (x(t_k) - x(t_k-1)) / (t_k - t_k-1)
    
    Args:
        t (array): Array of time value t_k, where k is the index of time
        x (array): Array of signal value x

    Returns:
        array: Array of calculated discrete derivative values
    
    Raises:
        ValueError: If length of t and x arrays are not equal
    
    Example:
        >>> arithmetic_progression([0, 0.1, 0.3, 0.4, 0.55, 0.67, 0.71], [23.1, 22.5, 23.5, 21.88, 22.5, 23.5, 24.88])
        [-6.000000000000014, 5.0, -16.200000000000003, 4.133333333333339, 8.333333333333334, 34.50000000000004]
    """
    # check equality of lengths of the two arrays
    if (len(t) != len(x)):
        raise ValueError("The input arrays are not of equal length")

    # compute the discrete derivative v(t), store values as python array
    v = [0 for _ in range(len(t)-1)] #v will be one value shorter than x and t
    for index in range(len(v)):
        v[index] = (x[index+1]-x[index])/(t[index+1]-t[index])

    # return v(t) array
    return v

# t = [0, 0.1, 0.3, 0.4, 0.55, 0.67, 0.71]
# x = [23.1, 22.5, 23.5, 21.88, 22.5, 23.5, 24.88]
# v = diff(t,x)
# print(v)