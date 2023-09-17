from flask import Flask, render_template, request
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt

app = Flask(__name__)

@app.route("/", methods=["GET", "POST"])
def index():
    if request.method == "POST":
        img = Image.open(request.files["file"])
        dominant_colors = get_dominant_colors(img)
        plot_colors(dominant_colors)
        return render_template("result.html")
    return render_template("upload.html")

def get_dominant_colors(img, n_colors=5):
    img = img.resize((100, 100))  # resizing to speed up processing time
    data = np.array(img)

    if data.shape[-1] == 4:  # If the image has an Alpha channel
        data = data.reshape(-1, 4)
        data = [tuple(item) for item in data if item[3] != 0]  # remove transparent pixels
    else:
        data = data.reshape(-1, 3)

    unique, counts = np.unique(data, axis=0, return_counts=True)
    sorted_indices = np.argsort(-counts)
    dominant_colors = unique[sorted_indices][:n_colors]

    return dominant_colors

def plot_colors(colors):
    fig, ax = plt.subplots(1, figsize=(5, 2), subplot_kw=dict(xticks=[], yticks=[], frame_on=False))
    for sp in ax.spines.values():
        sp.set_visible(False)
    ax.imshow([colors], aspect='auto')
    plt.savefig("static/output.png")

if __name__ == "__main__":
    app.run(debug=True)
