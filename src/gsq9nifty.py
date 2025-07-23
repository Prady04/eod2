import matplotlib.pyplot as plt
import numpy as np

# Starting from the last confirmed value
start = 24500
step = 34
end = 26200

# Generate the sequence
sequence = list(range(start, end + 1, step))

# Determine the best fit for square-like grid dimensions
cols = 9  # Gann Square is typically 9x9 or square-ish
rows = int(np.ceil(len(sequence) / cols))

# Pad the sequence to fill the grid
padded_sequence = sequence + [""] * (rows * cols - len(sequence))

# Reshape into grid
grid = np.array(padded_sequence).reshape(rows, cols)

# Plotting the grid
fig, ax = plt.subplots(figsize=(12, rows))  # Dynamic height based on rows
table = plt.table(cellText=grid, loc='center', cellLoc='center', colWidths=[0.1]*cols)
table.scale(1, 1.5)
ax.axis('off')
plt.title("Gann Square of 9 Style Grid (From 18258 to 25000, step=34)", fontsize=14)
plt.tight_layout()
plt.show()
plt.savefig('sqof9.jpg')