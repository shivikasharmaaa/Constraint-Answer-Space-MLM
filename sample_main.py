
from my_model import custom_model
answer_tokens = ['positive', 'negative']
c_tokens = [['positive','happy'],['negative','sad']]

model_no_map, tokenizer = custom_model('bert-base-uncased',answer_tokens)
model_map, tokenizer = custom_model('bert-base-uncased',answer_tokens, c_tokens)

sentence = "The text 'I am very happy today' has [MASK] sentiment."
inputs = tokenizer(sentence, return_tensors='pt', max_length=512, truncation=True, padding='max_length')

print("-----------------------------")
print("Output of model with simple answer tokens")
print(f"\tOutput of forward pass: {model_no_map(inputs, 1)}")
print(f"\tClass Token Probabilities: {model_no_map.return_probs(inputs)}")
print(f"\tPredicted Class Token: {model_no_map.return_prediction(inputs)}")

print("-----------------------------\n")
print("-----------------------------")
print("Output of model with mapped answer token set")
print(f"\tOutput of forward pass: {model_map(inputs, 1)}")
print(f"\tClass Token Probabilities: {model_map.return_probs(inputs)}")
print(f"\tPredicted Class Token: {model_map.return_prediction(inputs)}")
print("-----------------------------")