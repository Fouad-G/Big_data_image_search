import argparse
import sqlite3
import numpy as np
import pandas as pd
import plotly.express as px
from sklearn.manifold import TSNE

from imagerec.config import load_config
from imagerec.features.embedding import deserialize_embedding


def fetch_sample(db_path: str, dim: int, n: int) -> pd.DataFrame:
    conn = sqlite3.connect(db_path)
    rows = conn.execute(
        """
        SELECT e.image_id, e.vector, i.path
        FROM embeddings e
        JOIN images i ON i.id = e.image_id
        ORDER BY RANDOM()
        LIMIT ?
        """,
        (n,),
    ).fetchall()
    conn.close()

    vectors, paths, ids = [], [], []
    for image_id, blob, path in rows:
        vec = deserialize_embedding(blob, dim)
        vectors.append(vec)
        paths.append(path)
        ids.append(image_id)

    X = np.vstack(vectors)
    return X, pd.DataFrame({"image_id": ids, "path": paths})


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="config_full.yaml")
    ap.add_argument("--n", type=int, default=1000)
    ap.add_argument("--output", default="./data/full/interactive_map.html")
    args = ap.parse_args()

    cfg = load_config(args.config)

    X, meta = fetch_sample(cfg.db_path, cfg.embedding_dim, args.n)

    tsne = TSNE(n_components=2, perplexity=30, random_state=cfg.seed)
    Z = tsne.fit_transform(X)

    df = pd.DataFrame({
        "x": Z[:, 0],
        "y": Z[:, 1],
        "path": meta["path"]
    })

    fig = px.scatter(
        df,
        x="x",
        y="y",
        hover_data=["path"],
        title="Interactive Embedding Map (Click = Open Image)"
    )

    # Pfad als customdata speichern
    fig.update_traces(customdata=df["path"])

    # JavaScript: bei Klick Bild öffnen
    fig.update_layout(
        clickmode="event+select",
        annotations=[
            dict(
                text="Click on a point to open the image",
                x=0.5,
                y=1.05,
                xref="paper",
                yref="paper",
                showarrow=False,
            )
        ]
    )

    html = fig.to_html(include_plotlyjs="cdn")

    js = """
    <script>
    document.addEventListener("DOMContentLoaded", function() {
        var plot = document.querySelector(".plotly-graph-div");
        plot.on('plotly_click', function(data) {
            var path = data.points[0].customdata;
            window.open("file://" + path);
        });
    });
    </script>
    """

    html = html.replace("</body>", js + "</body>")

    with open(args.output, "w") as f:
        f.write(html)

    print("Saved:", args.output)


if __name__ == "__main__":
    main()
