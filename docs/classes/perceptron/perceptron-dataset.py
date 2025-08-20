import matplotlib.pyplot as plt
from io import StringIO

x1 = [1, 2, -1, -2]
x2 = [1, 2, -1, -1]

# Plot the data for the 2 firsts principal components
fig, ax = plt.subplots(figsize=(3, 3))

plt.xlabel('$x_1$')
plt.ylabel('$x_2$')

plt.plot(
    x1[:2], x2[:2], 'or',
    x1[2:], x2[2:], 'ob',
)
ax.set_xticks([])
ax.set_xticklabels([])
ax.set_yticks([])
ax.set_yticklabels([])

plt.tight_layout()

# Para imprimir na página HTML
buffer = StringIO()
plt.savefig(buffer, format="svg", transparent=True)
print(buffer.getvalue())