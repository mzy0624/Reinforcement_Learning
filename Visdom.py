import torch
import visdom

class Visdom:
    def __init__(self, env=''):
        self.vis = visdom.Visdom(env=env)
    
    def plot(self, x, y, win, name=None, **kwargs):
        self.vis.line(
            X=torch.tensor([x]),
            Y=torch.tensor([y]),
            win=win,
            name=name,
            update='append',
            opts=kwargs
        )