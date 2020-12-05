'''
Question 1 Skeleton Code

Here you should implement and experiment with the log-sum-exp function.
'''

import numpy as np
import warnings

def logsumexp_unstable(a, axis=None):
    '''
    Compute the logsumexp of the numpy array x along an axis in a numerically
    unstable way.

    Parameters
    ----------
    a : array_like
        Elements to logsumexp.
    axis : None or int or tuple of ints, optional
        Axis or axes along which a sum is performed.  The default,
        axis=None, will sum all of the elements of the input array.  If
        axis is negative it counts from the last to the first axis.
        If axis is a tuple of ints, a sum is performed on all of the axes
        specified in the tuple instead of a single axis or all the axes as
        before.

    Returns
    -------
    logsumexp_along_axis : ndarray
        An array with the same shape as `a`, with the specified
        axis removed.   If `a` is a 0-d array, or if `axis` is None, a scalar
        is returned.
    '''
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        logsumexp_along_axis = np.log(np.sum(np.exp(a), axis=axis))
    return logsumexp_along_axis

def logsumexp_stable(a, axis=None):
    '''
    Compute the logsumexp of the numpy array x along an axis.

    Parameters
    ----------
    a : array_like
        Elements to logsumexp.
    axis : None or int or tuple of ints, optional
        Axis or axes along which a sum is performed.  The default,
        axis=None, will sum all of the elements of the input array.  If
        axis is negative it counts from the last to the first axis.
        If axis is a tuple of ints, a sum is performed on all of the axes
        specified in the tuple instead of a single axis or all the axes as
        before.

    Returns
    -------
    logsumexp_along_axis : ndarray
        An array with the same shape as `a`, with the specified
        axis removed.   If `a` is a 0-d array, or if `axis` is None, a scalar
        is returned.  If an output array is specified, a reference to
        `out` is returned.
    '''
    m = np.max(a, axis=axis, keepdims=True)
    return np.log(np.sum(np.exp(a - m), axis=axis)) + np.squeeze(m, axis=axis)


def main():
    # Modify the elements of a so that logsumexp_unstable returns np.inf
    a = np.array([-1000, -1000, -1000])
    print(f"Computing the log-sum-exp of a={a}")
    print(f"\tUnstable: {logsumexp_unstable(a)}")
    print(f"\tStable: {logsumexp_stable(a)}")


    # Modify the elements of b so that logsumexp_unstable returns np.inf
    b = np.array([1000, 1000, 1000])
    print(f"Computing the log-sum-exp of b={b}")
    print(f"\tUnstable: {logsumexp_unstable(b)}")
    print(f"\tStable: {logsumexp_stable(b)}")

if __name__ == '__main__':
    main()
