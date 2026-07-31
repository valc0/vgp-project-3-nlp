# Fake vs. Real News Headline Classifier

An NLP project that classifies news headlines as **fake (`0`)** or **real (`1`)**, built for the Ironhack Data Science NLP Challenge.

## Task

Train a binary classifier on labeled news headlines to predict labels (`0`/`1`) for a test set marked with placeholder `2`s.

## Repository structure
```
.
├── code/
│   └── fake_news_classifier.ipynb
├── dataset/
│   ├── training_data.csv
│   ├── testing_data.csv
│   └── testing_data_predictions.csv
├── Model Comparison.png
├── project-3-nlp.pptx
└── README.md
```

## Data
| | Rows | Notes |
|---|---|---|
| Training set | 34,152 | 17,572 fake (`0`) · 16,580 real (`1`) — near-balanced |
| Test set | 9,984 | all labeled with placeholder `2` |

Both files are tab-separated with no header row: `label<TAB>headline`.

## Pipeline

1. **Cleaning:** Remove punctuation/artifacts, collapse whitespace, drop duplicates and label conflicts (reduces dataset to 32,197 rows).
2. **Feature Extraction:**
   - **TF-IDF:** Unigrams + bigrams without English stop words. Tested across linear, Naive Bayes, tree-based, distance-based, neural net models, and ensembles.
   - **Custom Embeddings:** Co-occurrence matrix → PPMI → 100D SVD → averaged headline vectors. Tested with Logistic Regression, Linear SVM, and MLP.
3. **Model Selection:** Candidates scored on an 85/15 stratified train/validation split by F1. Top model is chosen automatically.
4. **Final Predictions:** Winning model retrains on 100% of the data and outputs `testing_data_predictions.csv`.

## How to Run
Run `fake_news_classifier.ipynb` top to bottom. Generates `testing_data_predictions.csv` in the `/dataset` directory.

## Deliverables
- [x] Documented notebook (`fake_news_classifier.ipynb`)
- [x] Predictions CSV (`testing_data_predictions.csv`)
- [x] Results (`Model Comparison.png`)
- [x] Presentation (`project-3-nlp.pptx`)