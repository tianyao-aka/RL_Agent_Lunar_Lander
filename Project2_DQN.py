import gym
import numpy as np
import math
import random
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.utils.data import sampler
import torch.nn.functional as F
import torchvision.datasets as dset
import torchvision.transforms as T

USE_GPU = True

dtype = torch.float32 # we will be using float throughout this tutorial

if USE_GPU and torch.cuda.is_available():
    device = torch.device('cuda')
else:
    device = torch.device('cpu')

# Constant to control how frequently we print train loss

print('using device:', device)



env = gym.make('LunarLander-v2')
state=env.reset()
n_s=None
history = []
action=0
count = 0
step_count = 0
gamma = 0.9
eps_end = 0.005
eps_decay = 260000
N_capacity = 1e6
epsilon=0.99
episode = 1
state2, reward, done, _ = env.step(action)
state=torch.from_numpy(state).float()
state= state.view(1,-1)
memo_s = state
state2=torch.from_numpy(state2).float()
state2=state2.view(1,-1)
memo_ns = state2
memo_a = torch.tensor([action]).long()
memo_r = torch.tensor([reward]).float()
memo_done = torch.tensor([done]).long()
total_reward_arr= []
total_reward_calc= 0


class ThreeLayerFC(nn.Module):
    def __init__(self, input_size, hidden1_size,hidden2_size, hidden3_size,num_classes):
        super().__init__()
        # assign layer objects to class attributes
        self.fc1 = nn.Linear(input_size, hidden1_size)
        # nn.init package contains convenient initialization methods
        # http://pytorch.org/docs/master/nn.html#torch-nn-init
        nn.init.kaiming_normal_(self.fc1.weight)
        self.bn1=nn.BatchNorm1d(hidden1_size)
        self.fc2 = nn.Linear(hidden1_size, hidden2_size)
        nn.init.kaiming_normal_(self.fc2.weight)
        self.bn2 = nn.BatchNorm1d(hidden2_size)
        self.fc3 = nn.Linear(hidden2_size, hidden3_size)
        nn.init.kaiming_normal_(self.fc3.weight)
        self.bn3 = nn.BatchNorm1d(hidden3_size)
        self.fc4 = nn.Linear(hidden3_size,num_classes)
        nn.init.kaiming_normal_(self.fc4.weight)

    def forward(self, x):
        # forward always defines connectivity
        N= x.shape[0]
        x = x.view(N,-1)
        x = F.relu(self.bn1(self.fc1(x)))
        x = F.relu(self.bn2(self.fc2(x)))
        x = F.relu(self.bn3(self.fc3(x)))
        scores = self.fc4(x)
        return scores

def explore(model,state):
    global epsilon
    state = change_state(state)
    prob_explore = random.uniform(0,1)
    model.eval()
    with torch.no_grad():
        if prob_explore >= epsilon:
            action = torch.argmax(model(state),dim=1).item()
        else:

            max_action = torch.argmax(model(state),dim=1).item()
            while True:
                action = random.randint(0,3)
                if action != max_action: break

    model.train()
    return int(action)

def change_state(state):
    state = torch.from_numpy(state).float()
    state = state.view(1,-1)
    return state



def sample(model,state,memo_s,memo_ns,memo_a,memo_r,memo_done):
    global epsilon,episode,step_count,total_reward_calc,total_reward_arr,total_reward_calc
    episode_count = 0
    step_count +=1
    action = explore(model,state)
    state2, reward, done, _ = env.step(action)
    total_reward_calc += reward
    next_state = state2.copy()
    state,state2 = change_state(state),change_state(state2)
    memo_s = torch.cat((memo_s,state),dim=0)
    memo_ns = torch.cat((memo_ns,state2),dim=0)
    action = torch.tensor([action]).long()
    reward = torch.tensor([reward]).float()
    done = torch.tensor([done]).long()
    memo_a= torch.cat((memo_a,action)).long()
    memo_r= torch.cat((memo_r,reward)).float()
    memo_done= torch.cat((memo_done,done)).long()
    cur_length = len(memo_s)

    if done:
        env.reset()
        episode += 1
        step_count = 1
        total_reward_arr.append(total_reward_calc)
        total_reward_calc = 0
    if cur_length>=N_capacity:
        memo_s = memo_s[int(0.25*cur_length):]
        memo_ns = memo_ns[int(0.25*cur_length):]
        memo_a = memo_a[int(0.25*cur_length):]
        memo_r = memo_r[int(0.25*cur_length):]
        memo_done = memo_done[int(0.25*cur_length):]

    return (memo_s,memo_ns,memo_a,memo_r,memo_done,next_state)

def load_training_set(mini_batch=8196):
    cur_data_size = len(memo_s)
    idx = torch.randint(0,cur_data_size,(mini_batch,),dtype = torch.long)
    return memo_s[idx],memo_ns[idx],memo_a[idx],memo_r[idx],memo_done[idx]


def _get_training_score_for_init_train(target_model):
    with torch.no_grad():
        train_s,train_ns,train_a,train_r,train_done =  load_training_set()
        target_score= gamma*torch.max(target_model(train_ns),dim=1)[0]
        target_score += train_r
        target_score[train_done==True] = train_r[train_done==True]
        return target_score.detach(),train_s,train_a

def init_train(target_model,optimizer, num_iter=1):
    his=[]
    target_model = target_model.to(device=device)
    for e in range(num_iter):
        target_model.train()
        target_score,train_s,train_a= _get_training_score_for_init_train(target_model)
        batch_size = len(train_s)

        pred = target_model(train_s)
        pred = pred[torch.arange(batch_size).long(),train_a]

        loss = F.smooth_l1_loss(pred,target_score)


        # Zero out all of the gradients for the variables which the optimizer
        # will update.
        optimizer.zero_grad()

        # This is the backwards pass: compute the gradient of the loss with
        # respect to each  parameter of the model.
        loss.backward()

        # Actually update the parameters of the model using the gradients
        # computed by the backwards pass.
        optimizer.step()
        if e%10 == 0:
            print('Iteration %d, loss = %.4f' % (e, loss.item()))
            history.append(loss.item())
        #check_accuracy_part34(loader_val, model)
    return his


def train(target_model,cur_model, optimizer,history,num_iter=1,minibatch=256,gamma=0.9):
    """
    Train current model using Double-Q Learning

    """
    global memo_s,memo_ns,memo_a,memo_r,memo_done,n_s,count,epsilon,eps_end,eps_decay
    memo_s,memo_ns,memo_a,memo_r,memo_done,n_s = sample(cur_model,n_s,memo_s,memo_ns,memo_a,memo_r,memo_done)
    cur_model = cur_model.to(device=device)
    target_model = target_model.to(device=device)
    cur_model.train()
    for e in range(num_iter):
        train_s,train_ns,train_a,train_r,train_done =  load_training_set(minibatch)
        batch_size = len(train_s)
        with torch.no_grad():
            max_action_cur_model = torch.argmax(cur_model(train_ns),dim=1)
            target_score= target_model(train_ns)
            target_score = gamma*target_score[torch.arange(batch_size).long(),max_action_cur_model].detach()
            target_score += train_r
            #print ('target score',target_score)
            target_score[train_done==True] = train_r[train_done==True]

        pred = cur_model(train_s)
        #print ('pred before',pred.shape)
        pred = pred[torch.arange(batch_size).long(),train_a]

        loss = F.smooth_l1_loss(pred,target_score)

        optimizer.zero_grad()

        # This is the backwards pass: compute the gradient of the loss with
        # respect to each  parameter of the model.
        loss.backward()

        # Actually update the parameters of the model using the gradients
        # computed by the backwards pass.
        optimizer.step()
        count +=1
        epsilon = eps_end + (0.95 - eps_end)* math.exp(-1. * count / eps_decay)

        if count % 200 == 0:
            print('Iteration %d, loss = %.4f' % (count, loss.item()))
            if count % 2000 == 0: print ('episode:',episode,'epsilon:',epsilon)
            history.append(loss.item())
        for key in target_model.state_dict():
            update_params = target_model.state_dict()[key]*0.999 + cur_model.state_dict()[key]*0.001
            target_model.state_dict()[key].data.copy_(update_params)
            #print ('after',torch.sum(target_model.state_dict()[key]))


target_model = ThreeLayerFC(8,128,64,10,4)
cur_model = ThreeLayerFC(8,128,64,10,4)
n_s = env.reset()

env = gym.make('LunarLander-v2')
learning_rate = [8e-4]
gamma = [0.991]
eps_end = 0.06
episode = 1
history=[]
reward_list = []
qvalue_list= []
for lr in learning_rate:
    for g in gamma:
        print ('current parameters,learning rate:',lr,'gamma:',g)
        episode = 1
        count = 0
        test = True
        n_s = env.reset()
        target_model = ThreeLayerFC(8,128,64,10,4)
        cur_model = ThreeLayerFC(8,128,64,10,4)
        cur_model.load_state_dict(target_model.state_dict())
        for param in target_model.parameters():
            param.requires_grad = False
        target_model.eval()
        cur_model.train()
        for i in range(10000):
            memo_s,memo_ns,memo_a,memo_r,memo_done,n_s= sample(target_model,n_s,memo_s,memo_ns,memo_a,memo_r,memo_done)
        cur_optim = optim.Adam(cur_model.parameters(), lr=lr)
        eps_decay = 260000
        total_reward_arr= []
        total_reward_calc= 0
        while episode<=7002:

            train(target_model,cur_model, cur_optim, history,minibatch=256,gamma = g)

            if episode %50 == 0 and episode >=3000:
                print ('start to evaluate agent........')
                cur_model.eval()
                total=0
                total_q = 0
                test = False
                for i in range(10):
                    num=1
                    done = False
                    state= env.reset()
                    total_q += torch.max(cur_model(change_state(state))).item()
                    while done !=True:

                        action = torch.argmax(cur_model(change_state(state))).item()
                        state, reward, done, info = env.step(action)
                        total += reward
                        if done:
                            break
                reward_list.append(total/10.)
                qvalue_list.append(total_q/10.)
                cur_model.train()
                if total/10.>=200 and False:
                    torch.save(cur_model.state_dict(), 'DQN_lr_{}_gamma_{}_epi_{}_total_rewards_{}.pt'.format(lr,g,episode,str(total/10.0)[:6],total/10.))
                print ('currently in episode:',episode,' average reward:', total/10.0)

