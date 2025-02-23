## File Structure:

BERT.ipynb - Traning the BERT

SBERT.ipynb - Experiment version of SBERT with full dataset of SNLI(can't train using full dataset)

SBERT _experiment.ipynb - Use 1% of SNLI dataset

Note: Evaluation and Analysis, Inference and Web Application Development  are based on SBERT _experiment.ipynb 
## Evaluation and Analysis

| Epochs | Traning Loss | Validation Loss |
|----------|----------|----------|
| 1 | 2.01 | 2.00 |
| 2 | 1.77 | 1.81 |
| 3 | 1.29 | 1.14 |

# Inference

| sentence 1 | sentence 2 | Cosine Similarity |
|----------|----------|----------|
| "I love programming." | "Coding is my passion." | 0.9439 |



Classification Report:
| | precision | recall | f1-score |
|----------|----------|----------|----------|
|  entailment| 1.00 |  0.03  |  0.06   |
|   neutral |  0.28 | 0.29 |  0.29 |
|  contradiction|  0.34 | 0.68 | 0.46 |             
|  accuracy|  - | - | 0.33 | 
|  macro avg|  0.54  | 0.33 |  0.27 |
|   weighted avg |   0.55 | 0.33 |  0.26 |

|Accuracy: 0.3300 | Precision: 0.5539 | Recall: 0.3300 | F1-Score: 0.2629 |
|----------|----------|----------|----------|


| Limitations | Improvements |
|----------|----------|
|Training on the full SNLI dataset is computationally expensive and time-consuming |Use a smaller subset of the dataset (like 1%) for prototyping and debugging before scaling up to the full dataset |
| The model overfit to the training data, as the dataset is small  | Use regularization techniques: dropout,or early stopping |
| SNLI dataset may have imbalanced classes  | Use synonym replacement |

## Web Application Development
