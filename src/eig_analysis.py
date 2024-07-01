

import numpy as np


def PCA(data: np.ndarray, size: int) -> tuple[float, float]:
  
  height = width = size
  
  
  '''
  Perform Principal Component Analysis (PCA) on the provided data.

    PCA reduces the dimensionality of the data while preserving as much of the
    variance as possible. This is useful for reducing the number of dimensions
    in a dataset while still retaining as much information as possible.

    Parameters
    ----------
    data : np.ndarray
        The data matrix to perform PCA on, where each column represents a feature
        and each row represents an observation.
    size : int
        The desired number of dimensions to reduce the data to.

    Returns
    -------
    tuple[float, float]
        A tuple containing the eigenvalues and eigenvectors of the covariance matrix
        of the data, sorted in descending order. The first element of the tuple is
        the eigenvalues, and the second element is the eigenvectors.
  '''
  
  
  # PCA reduces dimensionality and preserves important features -> performance increase
  
  x = data.reshape(height * width)
  
  cov = x.T @ x / x.shape[0]
  
  eigval, eigvec = np.linalg.eig(cov)
  
  sorted = np.argsort(eigval)[::-1]
  
  return tuple(eigval[sorted], eigvec[:, sorted])