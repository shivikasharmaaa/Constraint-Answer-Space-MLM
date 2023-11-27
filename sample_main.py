
from my_model import custom_model

c_tokens = [['favor','favoring','preferred'],['against','opposing'],['none','neutral','unbiased']]
model, tokenizer = custom_model('bert-base-uncased')

sentence = "The tweet 'I hate Biden' has stance [MASK] towards Biden."
inputs = tokenizer(sentence, return_tensors='pt', max_length=512, truncation=True, padding='max_length')

print(model(inputs, 1))
print(model.return_probs(inputs))
print(model.return_prediction(inputs))