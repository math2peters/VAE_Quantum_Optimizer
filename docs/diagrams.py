import matplotlib.pyplot as plt
import matplotlib.patches as patches

# Create figure and axis
fig, ax = plt.subplots()

# Draw energy levels
y_positions = [0, 2]  # Energy levels for |0> and |1>
x_positions = [1]  # Energy level for |2>
labels = ['|0>', '|1>', '|2>']

for y, label in zip(y_positions, labels[:2]):
    plt.hlines(y, xmin=0, xmax=4, linewidth=2, color='black')
    plt.text(-0.5, y, label, verticalalignment='center')

plt.hlines(x_positions[0], xmin=5, xmax=9, linewidth=2, color='black')
plt.text(9.1, x_positions[0], labels[2], verticalalignment='center')

# Draw coupling arrow between |0> and |1>
arrow_coupling = patches.FancyArrowPatch((0.5, 0), (0.5, 2), connectionstyle="arc3,rad=.2", 
                                         arrowstyle='<->', mutation_scale=15, color='blue')
ax.add_patch(arrow_coupling)
plt.text(0.1, 1, 'f(t)', verticalalignment='center', color='blue')

# Draw decay line from |1> to |2>
arrow_decay = patches.FancyArrowPatch((3.5, 2), (6, 1), connectionstyle="arc3,rad=.2", 
                                      arrowstyle='->', mutation_scale=15, color='red')
ax.add_patch(arrow_decay)
plt.text(6.1, 1.4, r'$\Gamma$', verticalalignment='center', color='red', fontsize=16, fontweight='bold', style='italic')

# Set axis limits and hide axis
ax.set_xlim(-1, 10)
ax.set_ylim(-1, 3)
ax.axis('off')

plt.show()
