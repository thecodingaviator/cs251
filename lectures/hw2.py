import numpy as np

D = np.array( [[0.1,0.2,0.3],[0.3,0.4,0.5],[0.1,0.3,0.2],[0.6,0.7,0.8],[0.8,0.5,0.4],[0.9,0.1,0.2]] )

# Write the code to construct Dh, the data matrix with a column of ones (the homogeneous coordinate).
# Hint: use np.hstack() to stack the columns of ones and D together.
Dh = np.hstack((D, np.ones((D.shape[0],1))))

# Write the translation matrix that will translate the features by -0.5, 0.5, and 0.1 (in that order). Also, code it up. (you can include just code here)
T = np.eye(4)
T[:,3] = [-0.5, 0.5, 0.1, 1]

print(T)

# Write the code to transform the data matrix by the translation matrix. (you can include just code here)
Dtrfd = Dh @ T.T

# Scale features by 3,2,4 (in that order)
T = np.eye(4)
T[np.arange(3), np.arange(3)] = [3,2,4]

# Normalise D
S = np.eye(4)
S[np.arange(3), np.arange(3)] = 1 / ((np.max(D, axis=0) - np.min(D, axis=0)))

T = np.eye(4)
T[:3, 3] = -np.min(D, axis=0)



Dnorm =  ( S @ T @ Dh.T ).T

