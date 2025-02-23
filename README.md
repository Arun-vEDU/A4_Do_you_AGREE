## File Structure:

BERT - Traning the BERT
SBERT - Experiment version of SBERT with full dataset of SNLI

## Evaluation and Analysis

| Epochs | Traning Loss | Validation Loss |
|----------|----------|----------|
| 1 | 2.01 | 2.00 |
| 2 | 1.77 | 1.81 |
| 3 | 1.29 | 1.14 |

# Example sentences
sentence1 = "I love programming."

sentence2 = "Coding is my passion."

Cosine Similarity: 0.9439651966094971


| Limitations | Improvements |
|----------|----------|
|Training on the full SNLI dataset is computationally expensive and time-consuming |Use a smaller subset of the dataset (like 1%) for prototyping and debugging before scaling up to the full dataset |
| The model overfit to the training data, as the dataset is small  | Use regularization techniques: dropout,or early stopping |
| SNLI dataset may have imbalanced classes  | Use synonym replacement |
