## File Structure:

BERT - Traning the BERT
SBERT - Experiment version of SBERT with full dataset of SNLI

## Evaluation and Analysis

| Column 1 | Column 2 | Column 3 | Column 4 |
|----------|----------|----------|----------|
| Row 1, Col 1 | Row 1, Col 2 | Row 1, Col 3 | Row 1, Col 4 |
| Row 2, Col 1 | Row 2, Col 2 | Row 2, Col 3 | Row 2, Col 4 |


| Limitations | Improvements |
|----------|----------|
|Training on the full SNLI dataset is computationally expensive and time-consuming |Use a smaller subset of the dataset (like 1%) for prototyping and debugging before scaling up to the full dataset |
| The model overfit to the training data, as the dataset is small  | Use regularization techniques: dropout,or early stopping |
| SNLI dataset may have imbalanced classes  | Use synonym replacement |
