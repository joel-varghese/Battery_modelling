import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.linalg import hankel, svd

# Load your dataset
# Assuming your dataset is in a CSV file; update the path accordingly
data = pd.read_csv('/content/sample_data/Cap_curve.csv')

# Extract the 'capacity' column
capacity = data['capacity'].values

# Parameters
N = len(capacity)
L = N // 2  # Window length
K = N - L + 1

# Step 1: Construct the trajectory matrix (Hankel matrix)
X = hankel(capacity[:L], capacity[L-1:])

# Step 2: Singular Value Decomposition (SVD)
U, s, Vt = svd(X)
V = Vt.T

# Step 3: Reconstruct the time series using a selected number of components
d = 3  # Number of components to retain (you can change this)
X_reconstructed = np.dot(U[:, :d], np.dot(np.diag(s[:d]), Vt[:d, :]))

# Averaging over the anti-diagonals to reconstruct the time series
reconstructed_capacity = np.array([np.mean(np.diag(X_reconstructed[::-1], k)) for k in range(-X_reconstructed.shape[0] + 1, X_reconstructed.shape[1])])

# Add the denoised series back to the DataFrame
data['denoised_capacity'] = reconstructed_capacity

# Plot the original vs. denoised capacity
plt.figure(figsize=(10, 6))
plt.plot(data.index, capacity, label='Original Capacity')
plt.plot(data.index, reconstructed_capacity, label='Denoised Capacity', color='red')
plt.xlabel('Index')
plt.ylabel('Capacity')
plt.legend()
plt.title('Original vs. Denoised Capacity')
plt.show()

plt.savefig('denoising.png', format='png')

# Optionally, save the denoised data back to a CSV
data.to_csv('Denoise_cap_curve.csv', index=False)


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.linalg import hankel, svd

# Load your dataset
data = pd.read_csv('/content/sample_data/Cap_curve.csv')

# Extract the 'capacity' column
capacity = data['capacity'].values

# Parameters
N = len(capacity)
L = N // 2  # Window length
K = N - L + 1

# Step 1: Construct the trajectory matrix (Hankel matrix)
X = hankel(capacity[:L], capacity[L-1:])
plt.figure(figsize=(12, 10))

# Plot the trajectory matrix
plt.subplot(2, 2, 1)
plt.imshow(X, aspect='auto', cmap='viridis')
plt.colorbar()
plt.title('Trajectory Matrix (Hankel Matrix)')
plt.xlabel('Column Index')
plt.ylabel('Row Index')

# Step 2: Singular Value Decomposition (SVD)
U, s, Vt = svd(X)
V = Vt.T

# Plot singular values
plt.subplot(2, 2, 2)
plt.plot(s, 'o-', markersize=8)
plt.title('Singular Values')
plt.xlabel('Index')
plt.ylabel('Singular Value')


# Step 3: Reconstruct the time series using a selected number of components
d = 3  # Number of components to retain
X_reconstructed = np.dot(U[:, :d], np.dot(np.diag(s[:d]), Vt[:d, :]))

# Averaging over the anti-diagonals to reconstruct the time series
reconstructed_capacity = np.array([np.mean(np.diag(X_reconstructed[::-1], k)) for k in range(-X_reconstructed.shape[0] + 1, X_reconstructed.shape[1])])

# Plot the reconstructed trajectory matrix
plt.subplot(2, 2, 3)
plt.imshow(X_reconstructed, aspect='auto', cmap='viridis')
plt.colorbar()
plt.title('Reconstructed Trajectory Matrix')
plt.xlabel('Column Index')
plt.ylabel('Row Index')

# Plot the original vs. denoised capacity
plt.subplot(2, 2, 4)
plt.plot(data.index, capacity, label='Original Capacity')
plt.plot(data.index, reconstructed_capacity, label='Denoised Capacity', color='red')
plt.xlabel('Index')
plt.ylabel('Capacity')
plt.legend()
plt.title('Original vs. Denoised Capacity')

plt.tight_layout()
plt.show()

# Save the figure
plt.savefig('denoising.png', format='png')

# Optionally, save the denoised data back to a CSV
data['denoised_capacity'] = reconstructed_capacity
data.to_csv('Denoise_cap_curve.csv', index=False)
