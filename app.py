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

from sklearn.cluster import KMeans

def get_dominant_colors(img, n_colors=5):
    img = img.resize((100, 100))  # resizing to speed up processing time
    data = np.array(img)
    data = data[:, :, :3]  # Discarding the alpha channel if it exists
    data = data.reshape(-1, 3)
    
    kmeans = KMeans(n_clusters=n_colors)
    kmeans.fit(data)
    dominant_colors = kmeans.cluster_centers_
    
    return dominant_colors.astype(int)


def plot_colors(colors):
    fig, ax = plt.subplots(1, figsize=(5, 2), subplot_kw=dict(xticks=[], yticks=[], frame_on=False))
    for sp in ax.spines.values():
        sp.set_visible(False)
    ax.imshow([colors], aspect='auto')
    plt.savefig("static/output.png")

if __name__ == "__main__":
    app.run(debug=True)
