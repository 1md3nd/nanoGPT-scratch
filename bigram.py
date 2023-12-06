import time
import torch
import torch.nn as nn
from torch.nn import functional as F

batch_size = 32 # size of parallel batches of block_size (batch_dimension)
block_size = 8 # size of chunk of data we process (time_dimension)
max_iter = 3000
eval_interval = 500
learning_rate = 1e-2
eval_iter = 300

device = 'cuda' if torch.cuda.is_available() else 'cpu'
print(device)

with open('Rabindranath.txt','r',encoding='utf-8') as f:
    text = f.read()
chars = sorted(list(set(text)))
vocab_size = len(chars)

str2int = {ch:i for i,ch in enumerate(chars)}
int2str = {i:ch for i,ch in enumerate(chars)}
encode = lambda s:[str2int[c] for c in s]
decode = lambda l:''.join([int2str[n] for n in l])

data = torch.tensor(encode(text),dtype=torch.long)
n = int(0.9*len(data))
train_data = data[:n]
val_data = data[n:]


def get_batch(split):
    # to get random chunk of data for each training or validation
    data = train_data if split == 'train' else val_data
    idx = torch.randint(len(data)-block_size-1,(batch_size,))
    x = torch.stack([data[i:block_size+i] for i in idx])
    y = torch.stack([data[i+1:block_size+i+1] for i in idx])
    x,y = x.to(device),y.to(device)
    return x,y

@torch.no_grad()
def estimate_loss():
    losses = {}
    model.eval()
    for split in ['train','val']:
        loss = torch.zeros(eval_iter)
        for k in range(eval_iter):
            Xe, Ye = get_batch(split)
            _,lss = model(Xe,Ye)
            loss[k] = lss.item()
        losses[split] = loss.mean()
    model.train()
    return losses

class BigramModel(nn.Module):
    def __init__(self,vocab_size):
        super().__init__()
        # lookup table for tokens
        self.token_embedding_table = nn.Embedding(vocab_size,vocab_size)
        
    def forward(self,idx,target=None):
        # for each index in idx it reutrns token rows
        logits = self.token_embedding_table(idx) # return (B,T,C) (Batch,Time,Channel)
        if target is None:
            loss = None
        else:
            B,T,C = logits.shape
            logits = logits.view(B*T,C) # since cross_entropy accept differnt dime of input (B,C,T)
            target = target.view(B*T)
            loss = F.cross_entropy(logits,target)
        return logits,loss

    def generate(self,idx,max_tokens):
        for _ in range(max_tokens):
            logits,loss = self.forward(idx) # getting predictions
             # taking only the last idx prediction --> (B,C)
            logits = logits[:,-1,:] # (B,C)
            # softing for get probabilities
            probs = F.softmax(logits,dim=-1) # (B,C)
            # get one sample index from given probabilities and add to end of idx
            sample_idx = torch.multinomial(probs,num_samples=1)
            idx = torch.cat((idx,sample_idx),dim=1) # (B,T+1)
        return idx
    
model = BigramModel(vocab_size)
model.to(device)

# pytorch optimizer model AdamW
optimizer = torch.optim.AdamW(model.parameters(),lr=learning_rate)

st = time.time()
for curr_itr in range(max_iter):
    if curr_itr % eval_interval == 0:
        losses = estimate_loss()
        print(f"at {curr_itr} the train loss: {losses['train']:.4f} and val loss:{losses['val']:.4f}")
    xb,yb = get_batch('train')
    logits, loss = model(xb,yb)
    optimizer.zero_grad(set_to_none=True)
    loss.backward()
    optimizer.step()
et = time.time()
print('took',(et-st) % 60, 'seconds')
print('final loss',loss.item())

input_idx = torch.zeros((1,1), dtype = torch.long, device=device)
# generated tokens 
g_idx = model.generate(input_idx,200)

out = decode(g_idx[0].tolist())
print(out)