# Constraint-Answer-Space-MLM

## About
Often while using Masked Language Modeling for downstream tasks we want to restrict the answer space of our Language Model to some specified tokens (for example ['positive', 'negative'] for sentiment classification with the two tokens denoting separate classes). <br> <br>
This code (my_model.py) can be used to implement such a model. Here, the forward pass of the model is changed to accommodate the restrictions and the loss is calculated as the Cross-Entropy Loss between the probabilities of the restricted space and the ground truth label. <br> <br>
Furthermore, to enhance the prediction accuracy these class tokens can further be represented by a set of similar tokens (such as ['positive', 'happy', 'optimistic], etc. for the 'positive' sentiment class. This code can further accommodate a similar use case as well.

## Output of Sample Main
<img width="1022" alt="Screenshot 2023-11-28 at 10 43 14 AM" src="https://github.com/shivikasharmaaa/Constraint-Answer-Space-MLM/assets/91414321/a29931c0-4fd2-41e8-9aef-47050db55edb">


### Version Used
This code uses the libraries : 
- transformers  4.28.0
- torch         2.1.0
