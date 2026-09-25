# capstoneZeptoMasai
capstoneZeptoMasai  module wise

One repository with three connected modules:



## Setup

The project uses **one `requirements.txt` per module**:
[`data_pipeline/requirements.txt`](data_pipeline/requirements.txt),
[`analytics/requirements.txt`](analytics/requirements.txt) and
[`support_assistant/requirements.txt`](support_assistant/requirements.txt). This keeps the repository root
limited to this README and the three module folders. The project was developed and tested with
**Python 3.14** on Windows 11.

```bash
git clone https://github.com/nikhilkumarmuniganti-max/capstoneZeptoMasai.git
cd capstoneZeptoMasai

python -m venv .venv
# Windows:        .venv\Scripts\activate
# macOS / Linux:  source .venv/bin/activate

# everything at once, into one environment:
pip install -r data_pipeline/requirements.txt -r analytics/requirements.txt -r support_assistant/requirements.txt


```

## How to run each module



### 1. Data pipeline

```bash
cd data_pipeline
python run_pipeline.py                # scrape -> clean -> SQLite -> queries
python run_pipeline.py --skip-scrape  # offline: reuse the committed raw CSV
```

Results are written to [`data_pipeline/outputs/query_results.md`](data_pipeline/outputs/query_results.md).

### 2. Analytics

```bash
cd analytics
jupyter nbconvert --to notebook --execute --inplace 01_eda.ipynb       # loads Titanic once, saves titanic.csv
jupyter nbconvert --to notebook --execute --inplace 02_modeling.ipynb  # reads titanic.csv, trains, saves model
```

Run them in this order, or open them in Jupyter and choose "Run All". Both notebooks are committed with their
outputs, and every written interpretation, the model comparison table and the recommendation are in
[`analytics/README.md`](analytics/README.md).

### 3. Support assistant

_Coming next._

## Design decisions

### Data pipeline


### Analytics

_Coming next._

### Support assistant

_Coming next._

## Git workflow
