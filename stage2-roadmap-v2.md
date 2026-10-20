# Stage 2: Data and Machine Learning

Time: 2 to 3 months

## Table of Contents

- [Stage Goal](#stage-goal)
- [Phase 1: ML Workflow and Regression](#phase-1-ml-workflow-and-regression)
- [Phase 2: Classification Systems](#phase-2-classification-systems)
- [Phase 3: Trees, Ensembles, and Robust Evaluation](#phase-3-trees-ensembles-and-robust-evaluation)
- [Phase 4: Unsupervised Learning and Recommendation](#phase-4-unsupervised-learning-and-recommendation)
- [Stage 2 Milestone Challenge](#stage-2-milestone-challenge)
- [Complete Learning Timeline](#complete-learning-timeline)
- [Portfolio Checklist](#portfolio-checklist)
- [GitHub Project Checklist](#github-project-checklist)
- [Interview Preparation Checklist](#interview-preparation-checklist)
- [Common Interview Questions](#common-interview-questions)
- [Free Learning Resources](#free-learning-resources)
- [Books](#books)
- [Practice Websites](#practice-websites)
- [Advanced Topics Next](#advanced-topics-next)
- [Repository Structure](#repository-structure)
- [Phase README Teaching Template](#phase-readme-teaching-template)

## Stage Goal

Take a real dataset, clean and preprocess it, train suitable machine-learning models, evaluate them correctly, compare alternatives, explain limitations, and ship one complete application around the best model.

Prerequisites from Stage 1: Python, NumPy, Pandas, EDA, basic probability/statistics, vectors/matrices, derivatives, and gradient intuition.

Core tools: Python, scikit-learn, NumPy, Pandas, Matplotlib.

Primary sources:
- scikit-learn User Guide: https://scikit-learn.org/stable/user_guide.html
- scikit-learn Model Selection: https://scikit-learn.org/stable/model_selection.html
- scikit-learn Preprocessing: https://scikit-learn.org/stable/modules/preprocessing.html
- Pandas documentation: https://pandas.pydata.org/docs/
- NumPy documentation: https://numpy.org/doc/stable/
- Matplotlib documentation: https://matplotlib.org/stable/

Rule: learn the mathematics and system idea before using the estimator API. Never evaluate on training data and call that model performance.

## Phase 1: ML Workflow and Regression

### Phase Overview

**Goal:** Understand the complete machine-learning workflow using a prediction problem.

**Estimated study time:** 2 to 3 weeks.

**Prerequisites:** Stage 1 data/statistics foundations.

**Expected outcome:** Build a regression system from raw dataset to trustworthy evaluation.

### Concepts

- What machine learning is: learning a mapping from examples instead of manually writing every rule.
- Features `X` and target `y`.
- Training, validation, and test data.
- Data cleaning and preprocessing.
- Numeric scaling and categorical encoding.
- Feature engineering.
- Data leakage and why information from the future/test set must not enter training.
- Baseline models.
- Linear regression.
- K-nearest neighbors for regression.
- Underfitting and overfitting.
- Bias and variance intuition.
- Regression metrics: MAE, MSE, RMSE, R².
- scikit-learn workflow: `fit`, `predict`, transformers, `Pipeline`, train/test split.

**Math:** linear regression prediction, residual/error, mean squared error, distance intuition for KNN.

**Common mistakes:** preprocessing before splitting, using test data repeatedly, confusing feature with target, reporting one metric without context.

**Primary sources:**
- https://scikit-learn.org/stable/supervised_learning.html
- https://scikit-learn.org/stable/modules/linear_model.html
- https://scikit-learn.org/stable/modules/neighbors.html
- https://scikit-learn.org/stable/common_pitfalls.html
- https://scikit-learn.org/stable/modules/model_evaluation.html

### Practice

- Identify `X` and `y` from three datasets.
- Split a tiny dataset by hand into train/test sets.
- Find a leakage mistake in a broken workflow.
- Calculate MAE for a tiny prediction example by hand.
- Train a baseline, linear regression, and KNN regressor and compare them.

### Python Mini Project

**Prediction system:** predict a continuous value such as house price, delivery time, or energy use.

Why: it forces the complete workflow before classification adds more evaluation complexity.

### Architecture Decision

Backend: **No for the first implementation.** The learning target is the ML workflow, not network architecture.

Web/mobile: build only a thin demo after the Python model works. React and React Native collect feature values and display predictions. Do not duplicate model logic in JavaScript.

### Deployment

Deploy only the demo if useful. If serving live predictions, expose the trained model through a small Django REST Framework prediction endpoint after the Python pipeline is verified.

### Testing

- preprocessing tests
- model input-shape tests
- prediction smoke test
- leakage check
- metric calculation check

### Refactoring

Move preprocessing and model training into a reproducible `Pipeline`. Separate training, evaluation, and inference.

### Documentation

Dataset, target, features, cleaning decisions, split strategy, model choices, metrics, limitations, run instructions.

### Portfolio

Show the problem, EDA snapshot, baseline, model comparison, final metric, prediction demo, and limitations.

### Reflection

- Why do we split data?
- What is leakage?
- Why can low training error be misleading?
- Can I rebuild the pipeline without notes?

### Milestone Challenge

Given a new regression CSV, build a clean baseline and one improved model, evaluate both correctly, and explain which one you trust more and why.

## Phase 2: Classification Systems

### Phase Overview

**Goal:** Learn how models predict categories and how classification metrics change with the problem.

**Estimated study time:** 2 to 3 weeks.

**Prerequisites:** Phase 1 workflow and evaluation discipline.

**Expected outcome:** Build a classification system and justify its decision threshold and metrics.

### Concepts

- Binary and multiclass classification.
- Logistic regression and probability output.
- KNN classification.
- Support Vector Machines and margin intuition.
- Confusion matrix.
- Accuracy, precision, recall, F1.
- ROC curve and ROC-AUC.
- Class imbalance.
- Decision threshold.
- Cross-validation.
- Error analysis.

**Math:** sigmoid intuition, probability threshold, confusion-matrix metric formulas, distance, separating margin intuition.

**Common mistakes:** using accuracy on imbalanced data, tuning against the test set, assuming predicted probability is certainty.

**Primary sources:**
- https://scikit-learn.org/stable/modules/linear_model.html#logistic-regression
- https://scikit-learn.org/stable/modules/svm.html
- https://scikit-learn.org/stable/modules/model_evaluation.html
- https://scikit-learn.org/stable/modules/cross_validation.html

### Practice

- Calculate precision, recall, and F1 from a small confusion matrix.
- Explain when false positives or false negatives cost more.
- Compare logistic regression, KNN, and SVM on the same split.
- Inspect misclassified rows.

### Python Mini Project

**Classification system:** classify a customer, message, medical sample, or similar public dataset into categories.

Why: it makes evaluation choices concrete instead of treating accuracy as universal.

### Architecture Decision

Backend: **Yes for the finished demo** if predictions should be requested from web/mobile clients. Use Django REST Framework to load the saved model once and expose a prediction endpoint.

Web: React form, prediction result, confidence/probability where appropriate, and clear validation.

Mobile: React Native version of the same prediction flow.

Offline support: **No by default.** Model inference lives on the backend, so cached predictions would not replace live inference reliably.

### Deployment

Deploy API and web demo. Mobile can remain a development build unless publishing adds real portfolio value.

### Testing

- API validation tests
- inference tests
- metric tests
- frontend request/error tests
- malformed-input tests

### Refactoring

Create one reusable inference service. Keep model loading outside request-specific logic.

### Documentation

Include confusion matrix, chosen metrics, threshold, failure examples, and model limitations.

### Portfolio

Show why the chosen metric fits the problem, not just the final score.

### Reflection

- Why is accuracy sometimes dangerous?
- What changes when recall matters more than precision?
- Can I explain one wrong prediction?

### Milestone Challenge

Given an unfamiliar classification dataset, choose suitable metrics before training, compare at least two algorithms, perform error analysis, and defend the final choice.

## Phase 3: Trees, Ensembles, and Robust Evaluation

### Phase Overview

**Goal:** Learn nonlinear models and compare them without fooling yourself.

**Estimated study time:** 2 to 3 weeks.

**Prerequisites:** Phases 1 and 2.

**Expected outcome:** Build a strong fraud-risk style classifier and explain the tradeoffs of tree ensembles.

### Concepts

- Decision trees: splits, impurity, depth, leaves.
- Random forests: many decorrelated trees and aggregation.
- Gradient boosting: sequentially correcting previous errors.
- Hyperparameters.
- Validation strategy and cross-validation.
- Overfitting control.
- Feature importance and its limitations.
- Error analysis by subgroup.
- Imbalanced classification and fraud-detection thinking.

**Math:** impurity intuition, averaging/voting, residual-error correction intuition for boosting.

**Common mistakes:** unlimited tree depth, treating feature importance as causality, optimizing repeatedly on the final test set.

**Primary sources:**
- https://scikit-learn.org/stable/modules/tree.html
- https://scikit-learn.org/stable/modules/ensemble.html
- https://scikit-learn.org/stable/modules/generated/sklearn.model_selection.cross_validate.html

### Practice

- Manually choose a simple split for a tiny dataset.
- Compare shallow and deep trees.
- Compare decision tree, random forest, and gradient boosting.
- Run cross-validation and compare score variation.
- Inspect false negatives in an imbalanced problem.

### Python Mini Project

**Fraud detection system:** use a public transaction/fraud-style dataset or another strongly imbalanced classification dataset.

Why: it forces precision/recall tradeoffs, class imbalance, error analysis, and robust validation.

### Architecture Decision

Backend: **Yes.** Fraud-style inference is naturally a server-side decision service.

Web: React review dashboard showing transaction input, prediction, and model warning/limitations.

Mobile: React Native review screen if useful, kept intentionally small.

Offline support: **No.** Risk decisions should use the current deployed model and server-side controls.

### Deployment

Deploy the API and web dashboard. Keep secrets/config outside source control. Mobile publishing is optional.

### Testing

- model serialization/load test
- API schema/validation tests
- known-example regression tests
- imbalanced-metric tests
- frontend error-state tests

### Refactoring

Separate training artifacts from inference code. Version the model and preprocessing pipeline together.

### Documentation

Explain imbalance, chosen metric, threshold, false-positive/false-negative cost, validation method, limitations, and ethical risks.

### Portfolio

Emphasize evaluation discipline and failure analysis rather than claiming production fraud prevention.

### Reflection

- Why can a model with 99% accuracy still be useless?
- Why does a random forest usually overfit less than one deep tree?
- How is boosting different from bagging?

### Milestone Challenge

Take an imbalanced dataset, establish a baseline, train three tree-based models, use cross-validation, select metrics based on the problem, and produce an error-analysis report.

## Phase 4: Unsupervised Learning and Recommendation

### Phase Overview

**Goal:** Learn from data without target labels and build a practical recommendation system.

**Estimated study time:** 2 to 3 weeks.

**Prerequisites:** Phases 1 to 3.

**Expected outcome:** Cluster unlabeled data and build a recommendation system with honest evaluation and limitations.

### Concepts

- Supervised vs unsupervised learning.
- K-means clustering.
- Centroids and assignment/update cycle.
- Scaling before distance-based methods.
- Choosing `k` with domain reasoning and diagnostics.
- Cluster interpretation.
- Recommendation basics.
- Content-based recommendation using feature similarity.
- Similarity and nearest-neighbor retrieval.
- Recommendation limitations, cold start, feedback loops.

**Math:** Euclidean distance, centroid mean, nearest-neighbor similarity intuition.

**Common mistakes:** treating clusters as ground-truth classes, skipping scaling, claiming recommendations prove user preference.

**Primary sources:**
- https://scikit-learn.org/stable/modules/clustering.html#k-means
- https://scikit-learn.org/stable/modules/neighbors.html
- https://scikit-learn.org/stable/modules/preprocessing.html

### Practice

- Perform one K-means assignment step by hand on a tiny 2D example.
- Compare clustering before and after scaling.
- Explain what a cluster does and does not mean.
- Build nearest-item recommendations from numeric/categorical features.

### Python Mini Project

**Recommendation system:** recommend similar items from a public movies/products/books dataset, with K-means used separately for segmentation/exploration where useful.

Why: it connects unsupervised structure, preprocessing, similarity, and real product behavior without pretending K-means itself is a complete recommender.

### Architecture Decision

Backend: **Yes for the final Stage 2 application.** Django REST Framework serves items and recommendations.

Web: React browse/search page, item details, recommendations.

Mobile: React Native browse/details/recommendations.

Offline support: **Optional.** Cache recently viewed items and last recommendations only. Do not attempt complex bidirectional synchronization in this stage.

### Deployment

Deploy backend/API and web app. Mobile development build is enough unless publishing is worthwhile.

### Testing

- recommendation result-shape tests
- no-self-recommendation test
- API tests
- empty/unknown item tests
- UI loading/error tests

### Refactoring

Separate data preparation, training/index building, inference/retrieval, and API layers.

### Documentation

Explain data, features, similarity method, clustering role, recommendation limitations, cold-start behavior, and run/deploy steps.

### Portfolio

This is the main Stage 2 product. Include architecture diagram, screenshots, API examples, evaluation notes, and limitations.

### Reflection

- What can unsupervised learning discover without labels?
- Why must distance-based data often be scaled?
- Why is recommendation quality harder to measure than simple classification accuracy?

### Milestone Challenge

Given a new dataset, decide whether the problem is supervised or unsupervised, build the correct pipeline, evaluate it appropriately, and explain what conclusions the model cannot support.

## Stage 2 Milestone Challenge

Take one unfamiliar real dataset and complete the full workflow without tutorial code:

1. Define the problem and target, or explain why there is no target.
2. Clean and preprocess without leakage.
3. Establish a baseline.
4. Train at least three suitable models.
5. Use validation/cross-validation correctly.
6. Select appropriate metrics.
7. Perform error analysis.
8. Explain overfitting/underfitting and limitations.
9. Save a reproducible pipeline/model artifact.
10. Expose the final model through a small API only if the use case needs live inference.
11. Document and rebuild the core workflow from memory.

**Stage exit criterion:** Given a new small-to-medium tabular dataset, independently produce a defensible ML solution and explain why the evaluation can be trusted.

## Complete Learning Timeline

| Weeks | Focus |
| --- | --- |
| 1 to 3 | Phase 1: workflow, preprocessing, regression |
| 4 to 5 | Phase 2: classification and metrics |
| 6 to 8 | Phase 3: trees, ensembles, fraud-style evaluation |
| 9 to 10 | Phase 4: K-means and recommendation |
| 11 to 12 | Stage milestone, rebuilds, documentation, portfolio polish |

Move faster if exit criteria are genuinely passed. Do not stretch a phase merely to match the calendar.

## Portfolio Checklist

- [ ] Regression prediction system
- [ ] Classification system
- [ ] Fraud-detection style evaluation project
- [ ] Recommendation system
- [ ] At least one deployed ML API/web demo
- [ ] Model comparison and error-analysis reports
- [ ] Stage 2 milestone project

## GitHub Project Checklist

- [ ] Clear README
- [ ] Dataset source/license noted
- [ ] Reproducible environment/dependencies
- [ ] No dataset secrets or credentials committed
- [ ] Training and inference separated
- [ ] Saved preprocessing + model pipeline
- [ ] Tests
- [ ] Screenshots/demo
- [ ] Metrics and limitations
- [ ] Meaningful commits

## Interview Preparation Checklist

Be able to explain without notes:

- train vs validation vs test
- leakage
- overfitting vs underfitting
- bias vs variance
- regression vs classification
- linear vs logistic regression
- KNN
- decision tree vs random forest vs gradient boosting
- SVM intuition
- K-means
- accuracy, precision, recall, F1, ROC-AUC
- MAE, MSE, RMSE, R²
- cross-validation
- class imbalance
- feature preprocessing
- error analysis
- why a model can fail after deployment

## Common Interview Questions

1. Why should the test set stay untouched until the end?
2. What is data leakage? Give an example.
3. When is precision more important than recall?
4. When is recall more important than precision?
5. Why can accuracy be misleading?
6. What causes overfitting?
7. How does a random forest differ from a decision tree?
8. Bagging vs boosting?
9. Why does feature scaling matter for KNN, SVM, and K-means?
10. What does cross-validation tell you?
11. What does ROC-AUC measure?
12. MAE vs RMSE?
13. How would you debug a model that performs well in training but poorly in production?
14. What assumptions or limitations does your final project have?

## Free Learning Resources

- scikit-learn Getting Started: https://scikit-learn.org/stable/getting_started.html
- scikit-learn User Guide: https://scikit-learn.org/stable/user_guide.html
- Kaggle Learn: https://www.kaggle.com/learn
- Google Machine Learning Crash Course: https://developers.google.com/machine-learning/crash-course
- StatQuest: https://www.youtube.com/@statquest
- 3Blue1Brown for mathematical intuition: https://www.youtube.com/@3blue1brown

Use outside videos only when the concept benefits from a visual explanation. Return afterward and prove understanding through an exercise/project.

## Books

- *An Introduction to Statistical Learning*, James, Witten, Hastie, Tibshirani, Taylor
- *Hands-On Machine Learning with Scikit-Learn, Keras, and TensorFlow*, Aurélien Géron, use the classical ML sections first

## Practice Websites

- Kaggle datasets and notebooks: https://www.kaggle.com/
- UCI Machine Learning Repository: https://archive.ics.uci.edu/
- OpenML: https://www.openml.org/

## Advanced Topics Next

After Stage 2, move to deep learning only after the Stage 2 exit criterion is passed. Topics can include neural networks, PyTorch, optimization in deeper models, computer vision, NLP, and later applied LLM/AI engineering.

## Repository Structure

Keep Stage 2 inside the existing `ai-foundation` repository.

Recommended name:

```text
stage-2-data-and-machine-learning/
```

Recommended structure:

```text
ai-foundation/
├── stage-2-data-and-machine-learning/
│   ├── README.md
│   ├── phase-01-ml-workflow-and-regression/
│   │   ├── README.md
│   │   ├── exercises/
│   │   ├── project/
│   │   └── tests/
│   ├── phase-02-classification-systems/
│   │   ├── README.md
│   │   ├── exercises/
│   │   ├── project/
│   │   └── tests/
│   ├── phase-03-trees-ensembles-evaluation/
│   │   ├── README.md
│   │   ├── exercises/
│   │   ├── project/
│   │   └── tests/
│   ├── phase-04-unsupervised-and-recommendation/
│   │   ├── README.md
│   │   ├── exercises/
│   │   ├── project/
│   │   └── tests/
│   └── milestone/
│       ├── README.md
│       └── project/
```

Do not create separate phases just for individual algorithms. Algorithms belong inside the problem phase where they make sense.

## Phase README Teaching Template

At the start of every phase, create that phase's `README.md` using this outline so progress is visible and teaching stays disciplined:

```markdown
# Phase X: Title

## Goal

## Prerequisites

## Exit Criteria

## Teaching Order

- [ ] Concept 1
- [ ] Concept 2
- [ ] Concept 3

## Teaching Method for Every New Concept

1. Retrieval check for prerequisites already learned.
2. State what problem we are trying to solve.
3. Explain the real system in plain English.
4. Give one tiny concrete example.
5. Explain why the concept exists and what fails without it.
6. Give the math/intuition before library code when math is involved.
7. Provide the official primary source and one short exact quote.
8. Show one small implementation step.
9. Learner practices independently.
10. Diagnose mistakes with hints before giving the full solution.
11. Connect the concept to the phase project.
12. Record evidence in the learning-gap tracker.

## Exercises

- [ ] Hand/intuition exercise
- [ ] Coding exercise
- [ ] Debugging exercise
- [ ] Recall check

## Project

### Problem

### Dataset

### Architecture Decision

### Build Checklist

- [ ] Baseline
- [ ] Training pipeline
- [ ] Evaluation
- [ ] Error analysis
- [ ] Tests
- [ ] Refactor
- [ ] Documentation
- [ ] Deployment if useful

## Rebuild From Memory

- [ ] Explain the system without notes
- [ ] Rebuild the core pipeline without tutorial code
- [ ] Pass the phase milestone challenge

## Git Milestone

- [ ] `git status`
- [ ] `git add ...`
- [ ] meaningful commit
- [ ] `git push`
```
