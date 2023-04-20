from transformers import RobertaTokenizer, RobertaForQuestionAnswering, RobertaConfig
import torch
import time 
import csv

#path_token='/home/sharon/Documents/Research/HeterogeneousTaskScheduler/tokenizers/roberta-base/'
#path_model='/home/sharon/Documents/Research/HeterogeneousTaskScheduler/models/roberta-base.pt'
tokenizer = RobertaTokenizer.from_pretrained("roberta-base")
#tokenizer = RobertaTokenizer.from_pretrained(path_token, local_files_only=True)
#start=time.time()
#model = RobertaForQuestionAnswering.from_pretrained("roberta-base").cuda()
config = RobertaConfig.from_pretrained("roberta-base")
model= RobertaForQuestionAnswering(config).cuda()

#To save the model
#torch.save(model.state_dict(), '../models/roberta-base.pt') 



#state_dict=torch.load(path_model)
#model.load_state_dict(state_dict)
#end=time.time()
#cpu_task1_model1_loading=end-start
question, text = "Who was Jim Henson?", "Jim Henson was a nice puppet"
inputs = tokenizer(question, text, return_tensors="pt")
inputs.to('cuda:0')
start_positions = torch.tensor([1]).cuda()
end_positions = torch.tensor([3]).cuda()

#start=time.time()
outputs = model(**inputs, start_positions=start_positions, end_positions=end_positions)
#end=time.time()
#cpu_task1_model1_inference=end-start
print("done")
loss = outputs.loss
start_scores = outputs.start_logits
end_scores = outputs.end_logits
print(end_scores)


#with open('results.csv', 'a', newline='') as csvfile:
#    writer = csv.writer(csvfile, delimiter=',')
    #writer.writerow(['gpu','task_1','model_1','loading',gpu_task1_model1_loading])
#    writer.writerow(['cpu','task_1','model_1','loading',cpu_task1_model1_loading])
    #writer.writerow(['gpu','task_1','model_1','inference',gpu_task1_model1_inference])
#    writer.writerow(['cpu','task_1','model_1','inference',cpu_task1_model1_inference])
