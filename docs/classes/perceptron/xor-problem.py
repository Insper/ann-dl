import matplotlib.pyplot as plt
from io import StringIO

plt.rcParams["figure.figsize"] = (9, 3)

fig, ax = plt.subplots(1, 3)

for i in range(3):
  ax[i].axhline(0, color='gray') # x = 0
  ax[i].axvline(0, color='gray') # y = 0
  ax[i].spines['top'].set_visible(False)
  ax[i].spines['right'].set_visible(False)
  ax[i].spines['bottom'].set_visible(False)
  ax[i].spines['left'].set_visible(False)
  ax[i].set_xticks([0, 1])
  ax[i].set_yticks([0, 1])
  ax[i].set_xlim(-.1, 1.1)
  ax[i].set_ylim(-.1, 1.1)
  ax[i].set_xlabel('$x_1$')
  ax[i].xaxis.set_label_coords(.5, -.05)
  ax[i].set_ylabel('$x_2$')
  ax[i].yaxis.set_label_coords(-.05, .5)

i = 0
ax[i].title.set_text('AND')
ax[i].plot(0, 0, 'o', markersize=10, color='grey', markerfacecolor='white', markeredgecolor='red', markeredgewidth=2 )
ax[i].plot(0, 1, 'o', markersize=10, color='grey', markerfacecolor='white', markeredgecolor='red', markeredgewidth=2 )
ax[i].plot(1, 0, 'o', markersize=10, color='grey', markerfacecolor='white', markeredgecolor='red', markeredgewidth=2 )
ax[i].plot(1, 1, 'or', markersize=10)
ax[i].plot([.4, 1], [1.05, .5], '--b', lw=3)

i = 1
ax[i].title.set_text('OR')  
ax[i].plot(0, 0, 'o', markersize=10, color='grey', markerfacecolor='white', markeredgecolor='red', markeredgewidth=2 )
ax[i].plot(0, 1, 'or', markersize=10)
ax[i].plot(1, 0, 'or', markersize=10)
ax[i].plot(1, 1, 'or', markersize=10)
ax[i].plot([-.05, .6], [.6, -0.05], '--b', lw=3)

i = 2
ax[i].title.set_text('XOR')
ax[i].plot(0, 0, 'or', markersize=10)
ax[i].plot(0, 1, 'o', markersize=10, color='grey', markerfacecolor='white', markeredgecolor='red', markeredgewidth=2 )
ax[i].plot(1, 0, 'o', markersize=10, color='grey', markerfacecolor='white', markeredgecolor='red', markeredgewidth=2 )
ax[i].plot(1, 1, 'or', markersize=10)
ax[i].text(.5, 0.5, '?', horizontalalignment='center', fontsize=24)


# Adjust layout to prevent overlap
plt.tight_layout()

# Display the plot
buffer = StringIO()
plt.savefig(buffer, format="svg", transparent=True)
print(buffer.getvalue())
