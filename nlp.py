from transformers import DistilBertTokenizer, DistilBertForQuestionAnswering
import torch
import numpy as np

tokenizer = DistilBertTokenizer.from_pretrained("distilbert-base-uncased")
tokenizer.save_pretrained('../tokenizers/distilbert-base-uncased')

model = DistilBertForQuestionAnswering.from_pretrained("distilbert-base-uncased").cuda()
#torch.save(model.state_dict(), '../models/distilbert-base-uncased.pt')
print("done")
question, text = "Who was Jim Henson?", "Jim Henson was a nice puppet"
inputs = tokenizer(question, text, return_tensors="pt")
inputs.to('cuda:0')
start_positions = torch.tensor([1]).cuda()
end_positions = torch.tensor([3]).cuda()


outputs = model(**inputs, start_positions=start_positions, end_positions=end_positions)
loss = outputs.loss
start_scores = outputs.start_logits
end_scores = outputs.end_logits

possible_starts = np.argsort(start_scores.cpu().detach().numpy()).flatten()[::-1][0]
print(possible_starts)
print(inputs)

print(start_scores)

#print("Accuracy of BERT is:",accuracy_score(y_test,preds))

#import pdb
#pdb.set_trace()
#model.distilbert.transformer.layer[5].attention.v_lin.weight.sum()

#print(model)
#for name,param in model.named_parameters():
#    print(param)
#    print(name)
#print(outputs)
#print(end_scores)
