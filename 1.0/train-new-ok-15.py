from gensim.models import Word2Vec
import wandb
from gensim.models.callbacks import CallbackAny2Vec
import os
import pickle
import re
import multiprocessing
from tqdm.auto import tqdm

wandb.init(project="thai-w2v-new",
           config={
           })

data_all=[]
with open("save-cuted-ok.txt","r",encoding="utf-8-sig") as f:
    for i in tqdm(f): #len(new_train)
        data_all.append(eval(i))


# init callback class
class callback(CallbackAny2Vec):
    """
    Callback to print loss after each epoch
    """
    def __init__(self):
        self.epoch = 0
        self.best_loss = 1000000000
        self.loss_previous_step = 100000000000

    def on_epoch_end(self, model):
        loss = model.get_latest_training_loss()
        if self.epoch == 0:
            wandb.log({'epoch':self.epoch,'loss':loss,'loss_fix':loss})
        else:
            wandb.log({'epoch':self.epoch,'loss':loss,'loss_fix':loss- self.loss_previous_step})
        model.save(os.path.join("ok-15",'best-epoch-{}.bin'.format(self.epoch)))
        self.epoch += 1
        self.loss_previous_step = loss

model = Word2Vec(data_all, vector_size=400, window=15, min_count=5, workers=multiprocessing.cpu_count()-1,compute_loss=True,epochs=50,callbacks=[callback()])
wandb.finish()