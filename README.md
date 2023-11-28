# Constraint-Answer-Space-MLM

## About
Prompt Engineering has several important components such as Prompt Templates, Continuous or Discrete Prompts, Answer Spaces etc. <br>
Often while using Masked Language Modeling for downstream tasks we want to restrict the answer space of our Language Model to some specified tokens (for example ['positive', 'negative'] for sentiment classification with the two tokens denoting separate classes). <br> <br>
This code (my_model.py) can be used to implement such a model. Here, the forward pass of the model is changed to accommodate the restrictions and the loss is calculated as the Cross-Entropy Loss between the probabilities of the restricted space and the ground truth label. <br> <br>
Furthermore, to enhance the prediction accuracy these class tokens can further be represented by a set of similar tokens (such as ['positive', 'happy', 'optimistic], etc. for the 'positive' sentiment class. This code can further accommodate a similar use case as well.

## Output of Sample Main
<img width="957" alt="Screenshot 2023-11-28 at 12 34 56 PM" src="https://github.com/shivikasharmaaa/Constraint-Answer-Space-MLM/assets/91414321/2e6841e8-b323-4ea4-9b31-c5d7a71e8b45">



### Version Used
This code uses the libraries : 
- transformers  4.28.0
- torch         2.1.0

You can find my blog on restricted answer spaces in MLM [here](https://medium.com/@shivikas.29may/restricting-berts-answer-space-in-masked-language-modeling-d2cb2a8bdfd3)!
