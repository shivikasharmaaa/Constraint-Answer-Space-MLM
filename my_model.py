import torch
from transformers import BertForMaskedLM
import torch.nn as nn
from transformers import BertTokenizer
from transformers import AdamW
import random
import pickle

class My_BERT(nn.Module):
	def __init__(self, model_name, tokenizer, answer_tokens, c_tokens=None, is_map=False):
		super(My_BERT, self).__init__()
		self.BERT = BertForMaskedLM.from_pretrained(model_name)
		self.tokenizer = tokenizer
		for param in self.BERT.parameters():
			param.requires_grad = True

		self.answer_ids = self.tokenizer.encode(answer_tokens, add_special_tokens=False)
		self.N = len(answer_tokens)
		self.mask_token_id = 103
		self.loss_func = nn.CrossEntropyLoss()
		self.is_map = False
		if is_map:
			self.class_tokens = [self.tokenizer.encode(c_tk, add_special_tokens=False) for c_tk in c_tokens]
			self.is_map = True

	def forward(self, input_id, input_label):
		input_label = torch.tensor([input_label])
		outputs = self.BERT(input_ids=input_id['input_ids'],attention_mask=input_id['attention_mask'])
		out_logits = outputs.logits

		mask_position = input_id['input_ids'].eq(self.mask_token_id)
		mask_logits = out_logits[mask_position, :].view(out_logits.size(0), -1, out_logits.size(-1))[:, -1, :]

		answer_logits = mask_logits[:, self.answer_ids]
		if self.is_map:
			for c_index in range(len(self.class_tokens)):
				answer_logits[0][c_index] = torch.sum(mask_logits[:, self.class_tokens[c_index]])

		answer_probs = answer_logits.softmax(dim=1)

		loss = self.loss_func(answer_logits, input_label)

		return loss, answer_probs

	def return_probs(self, input_id):
		_, answer_probs = self.forward(input_id, random.randint(0,self.N-1))
		token_probs = zip(self.answer_ids, answer_probs[0])
		token_probs = [(self.tokenizer.convert_ids_to_tokens(ans_tk),ans_prob.item()) for ans_tk,ans_prob in token_probs]

		return sorted(token_probs, key=lambda x: x[1], reverse=True)

	def return_prediction(self, input_id):
		_, answer_probs = self.forward(input_id, random.randint(0,self.N-1))
		token_probs = zip(self.answer_ids, answer_probs[0])
		return self.tokenizer.convert_ids_to_tokens(max(token_probs, key=lambda x: x[1])[0])

def custom_model(model_name, c_tokens=None):
	tokenizer = BertTokenizer.from_pretrained(model_name)
	answer_tokens = ['favor', 'against', 'none']
	if c_tokens is not None:
		is_map = True
	else:
		is_map = False
	model = My_BERT(model_name, tokenizer, answer_tokens, c_tokens=c_tokens, is_map=is_map)

	return model, tokenizer
