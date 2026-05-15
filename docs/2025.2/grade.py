import matplotlib.pyplot as plt
from io import StringIO

fig, ax = plt.subplots(1, 2)

fig.set_size_inches(10, 5)

ax[0].pie(
    [15, 20, 35, 30],
    labels=["Data", "Perceptron", "MLP", "VAE"],
    colors=["mediumturquoise", "lightseagreen", "turquoise", "aquamarine"],
    autopct='%1.0f%%',
    startangle=90)
ax[0].title.set_text("Individual")

ax[1].pie(
    [30, 30, 40],
    labels=["Classification", "Regression", "Generative"],
    colors=["powderblue", "lightblue", "deepskyblue"],
    autopct='%1.0f%%',
    startangle=90)
ax[1].title.set_text("Team")

plt.tight_layout()

# Display the plot
buffer = StringIO()
plt.savefig(buffer, format="svg", transparent=True)
print(buffer.getvalue())

plt.close()