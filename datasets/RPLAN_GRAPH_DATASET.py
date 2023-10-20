from torch.utils import data
import os
from .floorplan_Loader import Load_floorplan_Plan_Graph
import random


class RPLAN_GRAPH_Dataset(data.Dataset):
    def __init__(self,path_root):
        self.floorplans=[os.path.join(path_root,pth) for pth in os.listdir(path_root)]
        random.shuffle(self.floorplans)
    def __len__(self):
        return len(self.floorplans)
    def __getitem__(self, index):
        the_pth=self.floorplans[index]
        floorplan=Load_floorplan_Plan_Graph(the_pth)
        input=floorplan.get_input()
        target=floorplan.get_output()
        return input,target

def build_RG_dataset(set_split,args):
    path=args.data_pth+set_split+"/"

    return RPLAN_GRAPH_Dataset(path)