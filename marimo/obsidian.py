import marimo

__generated_with = "0.17.2"
app = marimo.App(width="medium")


@app.cell
def _():
    import os
    import subprocess
    from datetime import datetime, timezone

    import calplot
    import matplotlib.pyplot as plt
    import pandas as pd
    from transformers import pipeline
    import marimo as mo
    return calplot, datetime, mo, os, pd, pipeline, plt, subprocess, timezone


@app.cell
def _(os, pipeline):
    # Sentiment Analysis
    #
    # Model:   https://huggingface.co/distilbert/distilbert-base-uncased-finetuned-sst-2-english
    # Dataset: https://huggingface.co/datasets/stanfordnlp/sst2
    #
    # - ⚠️ The data is very different from my notes, so the classifier seems heavily
    #      biased towards negative label.

    os.environ["TOKENIZERS_PARALLELISM"] = "false"

    model = "distilbert/distilbert-base-uncased-finetuned-sst-2-english"
    revision = "af0f99b"  # Newer version 714eb0f
    classifier = pipeline("sentiment-analysis", model=model, revision=revision)
    return (classifier,)


@app.cell
def _(classifier):
    # This should have been a neutral example, but is negative with high
    # confidence from this model. My notes have a lot of similar explainers, which
    # all end up 🟥.

    example = "I like RC"
    classifier(example)  # [{'label': 'POSITIVE', 'score': 0.9995481371879578}]
    return


@app.cell
def _(datetime, os, pd, subprocess, timezone):
    def ctime(root, path, filename):
        """Guess created timestamp from disk, git and filename"""

        sentinel = datetime.max.replace(tzinfo=timezone.utc)
        file_t, git_t, name_t = (sentinel, sentinel, sentinel)

        file_t = datetime.fromtimestamp(os.stat(path).st_ctime, tz=timezone.utc)

        # Creation time from the oldest commit that introduced the file
        cmd = [
            "git",
            "-C",
            root,
            "log",
            "-100",
            "--format=%ai",
            "--",
            path,
        ]
        try:
            git_ctime_str = subprocess.check_output(cmd).decode("utf-8").split("\n")[-1]
            git_t = datetime.fromisoformat(git_ctime_str)
        except ValueError:
            pass

        formats = ["%Y-%m-%d %H-%M", "%Y-%m-%d"]
        filename = filename.strip(".md").strip()
        for format in formats:
            try:
                # THIS TZ IS DEFINITELY SILLY
                name_t = datetime.strptime(filename, format).replace(tzinfo=timezone.utc)
            except ValueError:
                pass

        # print(f"{file_t=} \n{git_t=} \n{name_t=}")

        return pd.Timestamp(min([file_t, git_t, name_t]))
    return (ctime,)


@app.cell
def _(ctime, os):
    def read_md_files(directory: str):
        ignore = {".trash", ".archive", ".obsidian"}

        for root, _, files in os.walk(directory):
            if not any(i in root for i in ignore):
                for file in files:
                    if file.endswith(".md") and (not root.startswith(".")):
                        path = os.path.join(root, file)
                        with open(path, "r") as f:
                            yield (
                                {
                                    "name": file,
                                    "path": path,
                                    "contents": f.read(),
                                    "ctime": ctime(directory, path, file),
                                }
                            )
    return (read_md_files,)


@app.cell
def _(classifier, pd):
    def line_sentiment(line):
        sentiment = classifier(line)
        label = sentiment[0]["label"]
        score = sentiment[0]["score"]
        return score if label == "POSITIVE" else -score


    def sentiment(r):
        emotions = [line_sentiment(line[0:512]) for line in r.contents.split("\n")]
        emotions = pd.Series(emotions)
        happy = emotions[emotions > 0]
        sad = emotions[emotions < 0]
        return happy.prod() if happy.size > sad.size else sad.prod()
    return (sentiment,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""### Mongo Notes""")
    return


@app.cell
def _(pd, read_md_files):
    metadata = read_md_files("/Users/j/Notes/Mongo")
    df = pd.DataFrame(metadata)
    df = df.sort_values(by="ctime")
    df
    return (df,)


@app.cell
def _(df, sentiment):
    df["score"] = df.apply(sentiment, axis="columns")
    return


@app.cell
def _(df, pd):
    df["ctime"] = pd.DatetimeIndex(df["ctime"], tz="utc").tz_localize(None)
    values = df[["ctime", "score"]].set_index("ctime")
    values = values.resample("1d").mean()
    values
    return (values,)


@app.cell
def _(df, plt):
    plt.hist(df["score"].to_numpy(), 100)
    plt.title("Weights assigned by model")
    return


@app.cell
def _(calplot, values):
    calplot.calplot(
        values["score"], linewidth=0, cmap="RdYlGn", colorbar=False, dropzero=True
    )
    return


if __name__ == "__main__":
    app.run()
