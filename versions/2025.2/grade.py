import matplotlib.pyplot as plt
from io import StringIO

fig, ax = plt.subplots(1, 2)

fig.set_size_inches(10, 5)

ax[0].pie(
    [5, 5, 10, 5, 30, 45],
    labels=["Data", "Perceptron", "MLP", "Metrics", "Midterm", "Final"],
    explode=[0, 0, 0, 0, 0, 0],
    autopct='%1.0f%%',
    startangle=90)
ax[0].title.set_text("Individual")

ax[1].pie(
    [30, 30, 40],
    labels=["Classification", "Regression", "Generative"],
    explode=[0, 0, 0],
    autopct='%1.0f%%',
    startangle=90)
ax[1].title.set_text("Team")

plt.tight_layout()

# Display the plot
buffer = StringIO()
plt.savefig(buffer, format="svg", transparent=True)
print(buffer.getvalue())
