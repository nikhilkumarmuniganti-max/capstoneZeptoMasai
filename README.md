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

```bash
cd support_assistant
python check_assistant.py      # mock-mode acceptance checks (routing, retrieval, schema, no network)
uvicorn main:app --port 7860   # POST http://127.0.0.1:7860/ask  {"query": "..."}

docker build -t zepto-support-assistant .      # container alternative
docker run --rm -p 7860:7860 zepto-support-assistant
```


## Design decisions


### Data pipeline

- **Scope:** three whole categories (Mystery, Fantasy, Historical Fiction) giving 106 books. The scraper
  follows each category's "next" links, so pagination is handled rather than hard-coded.
- **Currency:** fixed baseline rate **1 GBP = 105.50 INR**, a project-defined constant, with no API call.
- **Messy rows:** unparseable numeric fields (price, rating) are median-imputed. Rows with unrecognised
  availability text, or with no title or category, are dropped. A built-in demo on deliberately broken rows
  shows the pipeline never crashes.
- **Schema:** `categories` (1) ──< `books` (many), linked by an enforced foreign key. The database is
  rebuilt from scratch on every run, so `database.py` is the exact recipe for `zepto_books.db`.
- **Verification:** the SQL JOIN result is rebuilt with `pd.merge` and compared automatically with
  `assert_frame_equal`.

Full details are in [`data_pipeline/README.md`](data_pipeline/README.md).

### Analytics

- **One load, one story:** `sns.load_dataset('titanic')` is called exactly once, in `01_eda.ipynb`, and
  immediately saved as `titanic.csv`. `02_modeling.ipynb` continues from that same file.
- **Missing values (threshold rule):**
  - `embarked` / `embark_town`, 0.22 % missing → drop those rows.
  - `age`, 19.87 % → impute with the median of each sex × class group.
  - `deck`, 77.22 % → keep it, with "Unknown" as its own category. The missingness is informative: survival
    is 29.9 % when the deck is unknown against 66.7 % when it is known.
- **Data story:** sex first, then class (with fare as its proxy), then young children and small families.
  This is shown through masked survival rates, a 6×6 correlation heatmap and five charts, each with a written
  interpretation.
- **No leakage:** a stratified 80/20 split (38 % survivors) comes first. All imputing, encoding and scaling
  lives in a `ColumnTransformer` inside a `Pipeline`, so it is fitted on the training split only, and inside
  each CV fold during `GridSearchCV`.
- **Models:** Logistic Regression, Decision Tree (depth 4) and Random Forest.
  - Imbalance: `class_weight='balanced'` raised recall from 0.69 to 0.75 at equal F1, so it was the best of
    baseline / balanced / SMOTE. SMOTE was applied only inside the training fold.
  - Tuning: GridSearchCV on the forest found `n_estimators=400, max_depth=8, max_features=None`, with an
    OOB score of 0.833.
- **Deployed model:** the tuned Random Forest, with test accuracy 0.831, precision 0.828 and F1 0.762. It is
  saved as one complete pipeline (`models/titanic_survival_pipeline.joblib`) that accepts raw passenger data.
- **Regression side task:** a linear model of fare reaches R² 0.347, and its residuals are clearly
  heteroscedastic.

### Support assistant

- **Mock LLM only:** with `MOCK_LLM` unset or `1`, every node answers from deterministic code. The real-LLM
  branch (`MOCK_LLM=0`, Groq), including its retry-on-invalid-JSON logic, is present in the code as the brief
  requires, but it was never enabled.
- **Pipeline:**
  - Ingestion: the 8 verbatim policy documents are split into sentence-level chunks (26 in total), so the top
    match is the exact rule asked about.
  - Embedding: local `all-MiniLM-L6-v2` vectors are stored in the ChromaDB collection `zepto_policies`
    (cosine).
  - Retrieval: the top-3 chunks, run for real in both modes.
  - Generation: the canned template `"Based on the retrieved context: …"`.
- **LangGraph:** a `TypedDict` state and three nodes (`classify_intent` → `retrieve_and_answer` |
  `direct_answer`) joined by a conditional edge. The intent is decided by the brief's exact 8-keyword
  heuristic.
- **Output contract:** a Pydantic `AskResponse` (`answer`, `sources` = retrieved chunk ids, `confidence` =
  top-chunk cosine similarity, or 1.0 for the canned general reply), served by FastAPI `POST /ask`.
- **Docker:** the image uses `python:3.12-slim`, CPU-only torch and a non-root user. The model is downloaded
  and the index built at build time, so the container runs offline on port 7860.

  finalized