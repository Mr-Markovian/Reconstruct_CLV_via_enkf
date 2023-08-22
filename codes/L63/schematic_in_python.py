import matplotlib.pyplot as plt
import numpy as np

# Define the domain
x = np.linspace(0, 2 * np.pi, 1000)
y = np.sin(x)

# Create the figure and axis
fig, ax = plt.subplots(figsize=(8, 6))

# Plot trajectory parts
ax.plot(x[0:250], y[0:250],lw=3, color='red', linestyle='dotted')
ax.plot(x[250:750], y[250:750],lw=3, color='black')
ax.plot(x[750:], y[750:],lw=3, color='blue', linestyle='dotted')

# Set plot title and labels
ax.set_xlabel('x')

ax.legend(handles=[
    plt.Line2D([], [], color='red', linestyle='dotted', label='$M_{j,j+1} B_j= B_{j+l} R_{j,j+l}$'),
    plt.Line2D([], [], color='black', label='$y = \sin(x)$'),
    plt.Line2D([], [], color='blue', linestyle='dotted', label='$C_j = R^{-1}_{j,j+l} C_{j+l}$')
], loc='upper right')

# Add legend
#ax.legend()

# Remove y labels and y axis
ax.set_yticklabels([])
ax.spines['left'].set_color('none')

# Remove top and right spines
ax.spines['top'].set_color('none')
ax.spines['right'].set_color('none')
# Show plot
plt.savefig('schematic.pdf',dpi=300)
