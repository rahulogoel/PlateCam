import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

def create_block(ax, coord, width, height, label, color, fontsize=12):
    x, y = coord
    rect = mpatches.FancyBboxPatch((x, y), width, height, boxstyle="round,pad=0.1", facecolor=color, edgecolor="black", linewidth=1)
    ax.add_patch(rect)
    ax.text(x + width / 2, y + height / 2, label, ha="center", va="center", fontsize=fontsize, color="white")

def add_arrow(ax, start, end):
    ax.annotate('', xy=end, xytext=start,
                arrowprops=dict(arrowstyle="->", lw=1.5))

# Set up the figure
fig, ax = plt.subplots(figsize=(18, 5))
ax.axis("off")

# Adjust block dimensions and spacing for a sleeker look
width = 3
height = 1
gap_x = 3.5

# Coordinates for blocks
x_start = 0
y = 0
coords = [
    (x_start, y),  # Collect Training Images
    (x_start + gap_x, y),  # Label Images for Training
    (x_start + 2 * gap_x, y),  # Train Deep Learning Model
    (x_start + 3 * gap_x, y),  # Save Model Artifacts
    (x_start + 4 * gap_x, y),  # Perform Object Detection & OCR
    (x_start + 5 * gap_x, y),  # Deploy Django Web Application
]

# Streamlined color palette and updated labels
labels = [
    "Collect Training Images",
    "Label Images for Training",
    "Train Deep Learning Model",
    "Save Model Artifacts",
    "Perform Object Detection & OCR",
    "Deploy Django Web Application",
]

colors = ["#4C72B0", "#55A868", "#C44E52", "#8172B3", "#CCB974", "#64B5CD"]

# Create streamlined blocks with adjusted font sizes
for i, coord in enumerate(coords):
    create_block(ax, coord, width, height, labels[i], colors[i], fontsize=10)

# Add arrows with adjusted sizes and alignment
for i in range(len(coords) - 1):
    start = (coords[i][0] + width, coords[i][1] + height / 2)
    end = (coords[i + 1][0], coords[i + 1][1] + height / 2)
    add_arrow(ax, start, end)

# Update bottom pipeline description to match the streamlined process
pipeline_steps = [
    "Data Collection",
    "Image Labeling",
    "Model Training",
    "Save Model",
    "OCR & Detection",
    "Django Deployment",
]
pipeline_x = [coords[i][0] + width / 2 for i in range(len(coords))]
pipeline_y = -1.5

for i, step in enumerate(pipeline_steps):
    ax.text(pipeline_x[i], pipeline_y, step, fontsize=9, ha="center", va="center", bbox=dict(boxstyle="round", facecolor="#F9F9F9", edgecolor="black"))

# Save the streamlined and professional flowchart
file_path_professional = "ANPR_Django_Flowchart_Professional.png"
plt.savefig(file_path_professional, bbox_inches="tight")
file_path_professional
