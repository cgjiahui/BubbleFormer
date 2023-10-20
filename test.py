from models import build_BubbleFormer
import argparse
from main import get_args_parser
import os
from datasets.floorplan_Loader import Load_floorplan_Plan_Graph
import numpy as np
import copy
import torch as t
import pickle
import graph_opt
color_map=np.array([
    [244, 241, 222],      #living room 0
    [234, 182, 159],    #bedroom  1
    [224, 122, 95],    #kitchen 2
    [95, 121, 123],    #bathroom 3
    [242, 204, 143],     #balcony 4
    [107, 112, 92],        #sotrage 5
    [100,100,100],      #exterior wall
    [255, 255, 25],     #FrontDoor
    [150,150,150], # interior wall
    [255,255,255]  #external
],dtype=np.int64
)

color_map[:,[0,2]]=color_map[:,[2,0]]






"Test and store for subsequent test"
def generate_bubble_graph(fp_pth, model, graph_number):
    "data"
    rooms_info_list = []
    connects_list = []
    for i in range(graph_number):
        Data_loader = Load_floorplan_Plan_Graph(fp_pth)
        Inputs = Data_loader.get_input()[:-5, :, :]


        noise = t.randn((1, 32, 30, 30))
        output = model.inference(Inputs.reshape((1, 3, 120, 120)), noise)

        rooms_info, connects = save_bubble_diagram(output)
        rooms_info_list.append(rooms_info)
        connects_list.append(connects)
    return rooms_info_list, connects_list

def save_bubble_diagram(outputs):
    rooms_info = []
    connects = []
    rooms_label = t.argmax(outputs['pred_logits'], -1)[0]
    for ind in range(8):
        if rooms_label[ind]!=6:
            pos = np.round(np.uint8(outputs['pred_points'].detach()[0][ind] * 120))[0:2]
            area = outputs['pred_points'][0][ind][2]
            one_room = {}
            one_room['pos'] = pos
            one_room['pixels'] = area.detach().numpy()
            one_room['category'] = rooms_label[ind].numpy()
            rooms_info.append(one_room)

            print(rooms_label)
            print(outputs['pred_logits'])
            print(outputs['pred_logits'][0][ind].detach().numpy())
        else:
            rooms_info.append(None)
    edge_types = np.argmax(outputs['pred_type'][0].detach().numpy(), -1)
    idx1 = np.argmax(outputs['pred_idx1'][0].detach().numpy(), -1)
    idx2 = np.argmax(outputs['pred_idx2'][0].detach().numpy(), -1)

    for i in range(16):
        if edge_types[i]:
            room_ind1 = idx1[i] if rooms_label[idx1[i]] != 6 else None
            room_ind2 = idx2[i] if rooms_label[idx2[i]] != 6 else None
            if (room_ind1!=None) and (room_ind2!=None) and (room_ind1!=room_ind2):
                connects.append(set([room_ind1, room_ind2]))
    connects_new = []
    for con in connects:
        if con not in connects_new:
            connects_new.append(con)
    connects_new = [list(con) for con in connects_new]
    return rooms_info, connects_new




def generate_bunch_bubble_diagrams(fp_pth, save_pth, model):
    file_name = os.path.splitext(os.path.split(fp_pth)[1])[0]
    with open(fp_pth, 'rb') as pkl_file:
        [wall_graph, boundary_graph, inter_graph, door_info, room_circles, rooms_info, connects, allG_iteration,
         new_window_mask] = pickle.load(pkl_file)

    gen_rooms_info, gen_connects = generate_bubble_graph(fp_pth, model, 100)
    gen_rooms_info.append(rooms_info)
    gen_connects.append(connects)

    save_file_pth = save_pth+f'{file_name}.pkl'
    save_file = open(save_file_pth, 'wb')
    pickle.dump([boundary_graph, gen_rooms_info, gen_connects, door_info], save_file, protocol=4)
    save_file.close()



def tes_return2(a,b):
    return a,b




if __name__ == '__main__':


    data_pth=""
    save_pth = './'
    model_pth = ""

    file_pths=[os.path.join(data_pth,single) for single in os.listdir(data_pth)]
    model_name = os.path.splitext(os.path.split(model_pth)[1])[0]
    "Load the model"
    parser = argparse.ArgumentParser('testing script', parents=[get_args_parser()])
    args = parser.parse_args()
    model, criterion = build_BubbleFormer(args)
    model.load_model(model_pth)
    model.eval()
    if not os.path.exists(save_pth):
        os.mkdir(save_pth)

    for one_pth in file_pths:
        generate_bunch_bubble_diagrams(one_pth, save_pth, model)



