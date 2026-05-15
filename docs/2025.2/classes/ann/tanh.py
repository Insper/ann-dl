import matplotlib.pyplot as plt
import numpy as np
from io import StringIO

x = np.arange(-1, 1, .1)
y = np.tanh(x)

fig, ax = plt.subplots()

fig.set_size_inches(8, 8)

ax.set_aspect(1.0)

plt.plot(x, y)

plt.xlim(0, 1.4)
plt.ylim(0, 1.4)
plt.axis('off')

# Adjust layout to prevent overlap
plt.tight_layout()

# Display the plot
buffer = StringIO()
plt.savefig(buffer, format="svg", transparent=True)
print(buffer.getvalue())

plt.close()