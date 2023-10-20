import copy
import cmath
import numpy as np
import torch as t
import random
import pickle
import cv2
import os
import graph_opt
pi=3.141592653589793


def area2r(area):
    return round(np.real(cmath.sqrt(area/pi)))

    "The data loader for the Transformer for only nodes"
class Load_floorplan_Plan_Graph():   #For floor generation

    "Use for loader for transformer"
    def __init__(self,pth):
        blur_pth = pth.replace("RPLAN_data_compact", "RPLAN_data_compact_blur")

        with open(blur_pth, 'rb') as pkl_blur_file:
            [inside_blur, outside_blur] = pickle.load(pkl_blur_file)


        with open(pth,'rb') as pkl_file:
            [wall_graph, boundary_graph, inter_graph, door_info, room_circles, rooms_info, connects, allG_iteration,
             new_window_mask]=pickle.load(pkl_file)

            "convert_size 256 2 120"
            wall_graph_120=graph_opt.convert_graph_256_120(wall_graph)
            boundary_graph_120=graph_opt.convert_graph_256_120(boundary_graph)
            door_info_120=graph_opt.convert_door_256_120(door_info)

            "prepare for the mask"
            boundary_mask_3pix=graph_opt.get_boundary_mask_120(boundary_graph_120,3)
            "inside mask get extracted from wall graph (some problem from boundary graph"
            inside_mask_120=graph_opt.get_inside_mask_120(wall_graph_120,boundary_mask_3pix)

            "door_mask"
            door_mask_120=graph_opt.get_door_mask_120(door_info_120)
            boun_door_mask=copy.deepcopy(boundary_mask_3pix)
            boun_door_mask[door_mask_120>0]=2


            "bubble diagram mask"
            bubble_node_mask, bubble_connect_mask, bubble_connect_liv_mask=self.get_bubble_mask(rooms_info,connects)
            "Input 2 torch"
            self.boundary_mask=t.from_numpy(boundary_mask_3pix).float()
            self.door_mask=t.from_numpy(door_mask_120).float()
            self.inside_mask=t.from_numpy(inside_mask_120).float()
            self.bubble_node_mask=t.from_numpy(bubble_node_mask).float()
            self.bubble_connect_mask=t.from_numpy(bubble_connect_mask).float()
            self.bubble_connect_liv_mask=t.from_numpy(bubble_connect_liv_mask).float()
            self.inside_blur_mask = t.from_numpy(inside_blur.astype(np.float16))
            self.outside_blur_mask = t.from_numpy(outside_blur.astype(np.float16))

            "Target 4 nodes & nodes label"
            rooms_num=len(rooms_info)
            rooms_pos=np.array([rooms_info[ind]['pos'] for ind in range(rooms_num)])
            rooms_pos=rooms_pos/120
            rooms_label=np.array([rooms_info[ind]['category'] for ind in range(rooms_num)])
            Total_area=0
            for one_room in rooms_info:
                Total_area+=one_room['pixels']
            rooms_area_ratio=np.array([[rooms_info[ind]['pixels']/Total_area] for ind in range(rooms_num)])

            rooms_triple=[]
            for ind in range(rooms_num):
                rooms_triple.append(np.concatenate((rooms_pos[ind],rooms_area_ratio[ind])))
            rooms_triple = np.array(rooms_triple)

            "Target 4 edges"
            room_pairs = []
            for ind in range(len(connects)):
                r_idx_1,r_idx_2=connects[ind]
                room_pairs.append(np.concatenate((rooms_triple[r_idx_1], rooms_triple[r_idx_2])))

            room_pairs = np.array(room_pairs)

            self.output_all={'labels':t.from_numpy(rooms_label),'points':t.from_numpy(rooms_triple), 'pair_nodes':t.from_numpy(room_pairs)}


    def get_bubble_mask(self, rooms_info, connects):
        bubble_node_mask = np.zeros((120, 120), dtype=np.uint8)
        bubble_connect_mask = np.zeros((120, 120), dtype=np.uint8)
        bubble_connect_liv_mask = np.zeros((120, 120), dtype=np.uint8)

        liv_ind=0
        for i in range(len(rooms_info)):
            one_room = rooms_info[i]
            if one_room['category'] == 0:
                liv_ind = i
            cv2.circle(bubble_node_mask, (one_room['pos'][1], one_room['pos'][0]), area2r(one_room['pixels']) // 3,
                       one_room['category'] + 2, -1)
        for con in connects:
            point1 = rooms_info[con[0]]['pos']
            point2 = rooms_info[con[1]]['pos']
            cv2.line(bubble_connect_mask, (point1[1], point1[0]), (point2[1], point2[0]), 2, 2, 8)
            if liv_ind in con:
                point1 = rooms_info[con[0]]['pos']
                point2 = rooms_info[con[1]]['pos']
                cv2.line(bubble_connect_liv_mask, (point1[1], point1[0]), (point2[1], point2[0]), 1, 2, 8)
                cv2.line(bubble_connect_mask, (point1[1], point1[0]), (point2[1], point2[0]), 1, 2, 8)

        return bubble_node_mask,bubble_connect_mask,bubble_connect_liv_mask

    def get_input(self):
        composite=t.zeros((8,120,120))
        composite[5] = self.bubble_node_mask
        composite[6] = self.bubble_connect_mask
        composite[7] = self.bubble_connect_liv_mask

        composite[0] = self.boundary_mask
        composite[1] = self.door_mask
        composite[2] = self.inside_mask
        composite[3] = self.inside_blur_mask
        composite[4] = self.outside_blur_mask
        return composite

    def get_output(self):
        return self.output_all

    def sample_BFS_Traversal(self,Graph,Iterations,button=0,process_split=[0.3,0.7]):
        input_nodes = []
        output_nodes = []
        last_nodes = []
        in_junction_mask = np.zeros((120, 120),dtype=np.uint8)
        in_wall_mask = np.zeros((120, 120),dtype=np.uint8)

        L=len(Iterations['iteration'])
        if button==0:
            iter_num=random.randint(1,L)

        for i in range(iter_num):
            input_nodes.extend(Iterations['iteration'][i])
        last_nodes.extend(Iterations['iteration'][iter_num-1])
        if iter_num<L:
            output_nodes.extend(Iterations['iteration'][iter_num])

        for ind in input_nodes:
            [c_h, c_w] = Graph[ind]['pos']
            in_junction_mask[c_h - 2:c_h + 3, c_w - 2:c_w + 3] = 1
            in_wall_mask[c_h - 1:c_h + 2, c_w - 1:c_w + 2] = 1
            for i in Graph[ind]['connect']:
                if i > 0 and i in input_nodes:
                    target = Graph[i]['pos']
                    cv2.line(in_wall_mask, (c_w, c_h), (target[1], target[0]), 1, 2, 4)

        for ind in last_nodes:
            [c_h, c_w] = Graph[ind]['pos']
            in_junction_mask[c_h - 2:c_h + 3, c_w - 2:c_w + 3] = 2
            in_wall_mask[c_h - 1:c_h + 2, c_w - 1:c_w + 2]=2
        return in_junction_mask,in_wall_mask,output_nodes





if __name__ == '__main__':
    fp_pth="F:/sjh_study/gs_next_works/RPLAN_data_compact/val/"
    pths=[os.path.join(fp_pth,single) for single in os.listdir(fp_pth)]

    for single in pths:
        fp_loader=Load_floorplan_Plan_Graph(single)
        outputs=fp_loader.get_output()
        print(outputs)
