import numpy as np
import numpy as np
from sklearn.linear_model import LinearRegression
import warnings


"""
This code is adapted from the R code of Jones (2019) available at:
https://github.com/TommyJones/mvrsquared
"""



def handle_y(y):
    if not isinstance(y, (np.ndarray, list)):
        raise ValueError("'y' must be a numeric vector or matrix.")

    if isinstance(y, list):
        y = np.array(y)

    if not np.issubdtype(y.dtype, np.number):
        raise ValueError("'y' does not appear to be numeric.")

    if y.ndim == 1:
        Y = y.reshape(-1, 1)
    elif y.ndim == 2:
        Y = y
    else:
        raise ValueError("'y' must be a 1D or 2D numeric array.")

    return Y

def handle_yhat(yhat, dim_y):
    if not isinstance(yhat, (np.ndarray, list, dict)):
        raise ValueError(
            "'yhat' must be a numeric vector, a numeric matrix, or a list/dict of two numeric matrices whose dot product makes the predicted value of 'y'."
        )

    if isinstance(yhat, np.ndarray) and yhat.ndim == 1:
        if dim_y[1] != 1:
            raise ValueError(
                "'yhat' does not have the correct number of columns to match 'y'."
            )
        if dim_y[0] != len(yhat):
            raise ValueError(
                "'yhat' does not have the correct number of observations to match 'y'."
            )
        x = yhat.reshape(-1, 1)
        w = np.ones((1, 1))

    elif isinstance(yhat, np.ndarray):
        if yhat.shape != dim_y:
            raise ValueError(
                f"'y' and 'yhat' do not have the same dimensions. 'y' has shape {dim_y}, but 'yhat' has shape {yhat.shape}"
            )
        x = yhat
        w = np.eye(yhat.shape[1])

    elif isinstance(yhat, (list, dict)):
        if isinstance(yhat, dict):
            if 'x' in yhat and 'w' in yhat:
                x = yhat['x']
                w = yhat['w']
            else:
                raise ValueError(
                    "When 'yhat' is a dict, it must contain 'x' and 'w' keys."
                )
        else:
            if len(yhat) < 2:
                raise ValueError(
                    "'yhat' must be a list containing at least two matrices."
                )
            x, w = yhat[:2]

        if not (isinstance(x, np.ndarray) and isinstance(w, np.ndarray)):
            raise ValueError(
                "Both elements of 'yhat' must be numpy arrays (matrices)."
            )

        if x.shape[0] != dim_y[0] or w.shape[1] != dim_y[1] or x.shape[1] != w.shape[0]:
            raise ValueError(
                "Dimensions of 'x' and 'w' in 'yhat' are not compatible with 'y'."
            )
    else:
        raise ValueError(
            "'yhat' must be a numeric vector, a numeric matrix, or a list/dict of two numeric matrices."
        )

    return {'x': x, 'w': w}

def calc_sum_squares_latent(Y, X, W, ybar, threads=1):
    Yhat = X @ W
    E = Y - Yhat
    SSE = np.sum(E ** 2)
    T = Y - ybar
    SST = np.sum(T ** 2)
    return [SSE, SST]

def calc_rsquared(y, yhat, ybar=None, return_ss_only=False, threads=1):
    if not isinstance(return_ss_only, bool):
        raise ValueError("'return_ss_only' must be boolean True/False.")

    if ybar is None and return_ss_only:
        warnings.warn(
            "'return_ss_only' is True but 'ybar' is None. If calculating in batches, you may get a misleading result. If not, disregard this warning."
        )

    Y = handle_y(y)
    Yhat = handle_yhat(yhat, Y.shape)

    if ybar is not None:
        if len(ybar) != Y.shape[1]:
            raise ValueError(
                "'ybar' is the wrong size. It must have the same length as the number of columns in 'y'."
            )
    else:
        ybar = Y.mean(axis=0)

    result = calc_sum_squares_latent(
        Y=Y, X=Yhat['x'], W=Yhat['w'], ybar=ybar, threads=threads
    )

    if return_ss_only:
        return {'sse': result[0], 'sst': result[1]}
    else:
        out = 1 - result[0] / result[1]
        return out


def jones(og, added):

    model = LinearRegression().fit(og, added)
    yhat = model.predict(og)

    # Calculate R-squared
    return calc_rsquared(added, yhat)
