import os
import numpy as np
import cv2
import copy
import sys
import random
import shutil
import pickle
color_map=np.array([    #https://coolors.co/f4f1de-e07a5f-3d405b-81b29a-f2cc8f
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
pi=3.141592653589793


def convert_graph_256_120(graph_256):
    graph_120=copy.deepcopy(graph_256)
    for i in range(len(graph_120)):
        if graph_120[i]!=None:
            graph_120[i]['pos'][0]=graph_120[i]['pos'][0]-8
            graph_120[i]['pos'][1] = graph_120[i]['pos'][1] - 8
            graph_120[i]['pos'][0] = graph_120[i]['pos'][0]//2
            graph_120[i]['pos'][1] = graph_120[i]['pos'][1]//2
    return graph_120

def convert_graph_256_512(graph_256):
    graph_512 = copy.deepcopy(graph_256)
    for i in range(len(graph_512)):
        if graph_512[i] != None:
            graph_512[i]['pos'][0] = graph_512[i]['pos'][0] * 2
            graph_512[i]['pos'][1] = graph_512[i]['pos'][1] * 2
    return graph_512

def convert_door_256_512(door_info):
    door_info_512 = copy.deepcopy(door_info)
    door_info_512['pos'] = np.array(door_info_512['pos'])*2
    return  door_info_512


def get_boundary_mask_512(boundary_graph, pixel_len):
    boundary_mask = np.ones((512, 512), dtype=np.uint8) * 255
    node_len = pixel_len // 2

    if pixel_len >= 5:
        line_len = pixel_len - 2
    elif pixel_len == 3:
        line_len = 2
    else:
        raise ValueError("The unsurpported wall length in mask")

    for node in boundary_graph:
        if node != None:
            ori = node['pos']
            boundary_mask[ori[0] - node_len:ori[0] + node_len + 1, ori[1] - node_len:ori[1] + node_len + 1] = 100
            for i in node['connect']:
                if i > 0:
                    target = boundary_graph[i]['pos']
                    cv2.line(boundary_mask, (ori[1], ori[0]), (target[1], target[0]), 100, line_len, 4)
    return boundary_mask

def add_door(boundary_mask_rgb, door_info_512, color):
    boundary_copy = copy.deepcopy(boundary_mask_rgb)
    pos = door_info_512['pos']
    if door_info_512['ori'] == 0:
        boundary_copy[pos[0]-3:pos[0]+4, pos[1]-10:pos[1]+10] = color
    else:
        boundary_copy[pos[0]-10:pos[0]+10, pos[1]-3:pos[1]+4] = color
    return boundary_copy

def render_bubble_diagram_mask(mask, rooms_info, connects):
    mask_copy = copy.deepcopy(mask)
    for one_room in rooms_info:
        if one_room !=None:
            pos_120 = one_room['pos']
            pos_512 = [(pos + 7) * 4 for pos in pos_120]
            color = color_map[one_room['category']]
            color = tuple([int(x) for x in color])
            r_512 = int(pow(one_room['pixels']*4*120*120*0.58 / pi, 0.5))

            "render the mask"
            cv2.circle(mask_copy, tuple([pos_512[1], pos_512[0]]), r_512, color, -1)

    for con in connects:
        pos1_512 = [(pos+7)*4 for pos in rooms_info[con[0]]['pos']]
        pos2_512 = [(pos+7)*4 for pos in rooms_info[con[1]]['pos']]

        cv2.line(mask_copy, tuple([pos1_512[1], pos1_512[0]]), tuple([pos2_512[1], pos2_512[0]]), [100, 100, 100], 4)

    return mask_copy

def render_bubble_diagram_mask_256(mask, rooms_info, connects):
    mask_copy = copy.deepcopy(mask)
    for one_room in rooms_info:
        if one_room !=None:
            pos_120 = one_room['pos']
            pos_512 = [(pos + 7) * 2 for pos in pos_120]
            color = color_map[one_room['category']]
            color = tuple([int(x) for x in color])
            r_512 = int(pow(one_room['pixels']*4 / (pi*3), 0.5))

            "render the mask"
            cv2.circle(mask_copy, tuple([pos_512[1], pos_512[0]]), r_512, color, -1)

    for con in connects:
        pos1_512 = [(pos+7)*2 for pos in rooms_info[con[0]]['pos']]
        pos2_512 = [(pos+7)*2 for pos in rooms_info[con[1]]['pos']]

        cv2.line(mask_copy, tuple([pos1_512[1], pos1_512[0]]), tuple([pos2_512[1], pos2_512[0]]), [100, 100, 100], 4)

    return mask_copy



def show_graph(graph,size):
    mask=np.zeros((size,size),dtype=np.uint8)
    for node in graph:
        if node != None:
            ori = node['pos']
            for i in node['connect']:
                if i > 0:
                    target = graph[i]['pos']
                    cv2.line(mask, (ori[1], ori[0]), (target[1], target[0]), 150, 2, 4)

    for node in graph:
        if node != None:
            ori = node['pos']
            mask[ori[0]-1:ori[0]+2,ori[1]-1:ori[1]+2]=255
    cv2.imshow("graph_256",mask)

def get_door_mask_120(door_info):
    door_mask=np.zeros((120,120),dtype=np.uint8)
    pos=door_info['pos']
    door_long=9//2
    if door_info['ori']==0:
        door_mask[pos[0]-2:pos[0]+3,pos[1]-door_long:pos[1]+door_long+1]=1
    else:
        door_mask[pos[0]-door_long:pos[0]+door_long+1,pos[1]-2:pos[1]+3]=1
    return door_mask

def convert_door_256_120(door_info_256):
    door_info_120=copy.deepcopy(door_info_256)
    door_info_120['pos']=(np.array(door_info_120['pos'])-8)//2
    return door_info_120

def get_boundary_mask_120(boundary_graph,pixel_len):
    boundary_mask = np.zeros((120, 120), dtype=np.uint8)
    node_len=pixel_len//2

    if pixel_len>=5:
        line_len=pixel_len-2
    elif pixel_len==3:
        line_len=2
    else:
        raise ValueError("The unsurpported wall length in mask")
    for node in boundary_graph:
        if node != None:
            ori = node['pos']
            boundary_mask[ori[0] - node_len:ori[0] + node_len+1, ori[1] - node_len:ori[1] + node_len+1] = 1
            for i in node['connect']:
                if i > 0:
                    target = boundary_graph[i]['pos']
                    cv2.line(boundary_mask, (ori[1], ori[0]), (target[1], target[0]), 1, line_len, 4)
    return boundary_mask

def show_some_nodes(Graph,nodes,size=120):
    mask=np.zeros((size,size),dtype=np.uint8)
    for ind in nodes:
        pos=Graph[ind]['pos']
        mask[pos[0]-2:pos[0]+3,pos[1]-2:pos[1]+3]=200
    cv2.imshow("nodes_mask",mask)


def get_inside_mask_120(boundary_graph,boundary_mask_120):
    inside_mask_120=np.zeros((120,120),dtype=np.uint8)
    points_array=[]
    boun_slices,boun_searched=Rotate_border_circle(boundary_graph)
    # print(boun_searched)
    for ind in boun_searched:
        points_array.append([boundary_graph[ind]['pos'][1],boundary_graph[ind]['pos'][0]])
    cv2.fillPoly(inside_mask_120,np.array([points_array]),1)
    inside_mask_120[boundary_mask_120>0]=0
    return inside_mask_120
def Rotate_border_circle(boundary_graph):
    start = 1
    slices = []
    searched = []
    searched.append(start)
    current = start
    next_node = boundary_graph[1]['connect'][3]
    break_times = 0
    while (next_node > 0):
        break_times = break_times + 1
        ori, distance = get_dire_d(boundary_graph, current, next_node)
        slices.append([ori, [current, next_node], distance])
        searched.append(next_node)
        current = next_node
        next_node = get_next_out(boundary_graph, start, current, ori)

        if break_times > 100:
            return slices, searched
    return slices, searched
def get_dire_d(junction_graph, junc1, junc2):
    if junction_graph[junc1]['pos'][0] == junction_graph[junc2]['pos'][0]:
        distance = junction_graph[junc1]['pos'][1] - junction_graph[junc2]['pos'][1]
        if distance > 0:
            return 2, abs(distance)
        elif distance < 0:
            return 3, abs(distance)
        else:
            print("提取墙段，获取两节点方向时出现错误，推出程序2")
            sys.exit(0)
    elif junction_graph[junc1]['pos'][1] == junction_graph[junc2]['pos'][1]:
        distance = junction_graph[junc1]['pos'][0] - junction_graph[junc2]['pos'][0]
        if distance > 0:
            return 0, abs(distance)
        elif distance < 0:
            return 1, abs(distance)
        else:
            print("提取墙段，获取两节点方向时出现错误，推出程序2")
            sys.exit(0)
    else:
        print("提取墙段，获取两节点方向时出现错误，推出程序1")
        sys.exit(0)
def get_next_out(junction_graph, start_node, current_node, current_ori):
    junction_graph_copy = copy.deepcopy(junction_graph)
    start_pos = junction_graph_copy[start_node]['pos']
    current_pos = junction_graph_copy[current_node]['pos']
    current_con = junction_graph_copy[current_node]['connect']
    current_con[get_reverse_ori(current_ori)] = 0
    if start_pos[0] == current_pos[0] and start_pos[1] < current_pos[1]:
        if current_ori == 3 or current_ori == 1:
            if current_con[0]:
                return current_con[0]
            elif current_con[3]:
                return current_con[3]
            elif current_con[1]:
                return current_con[1]
            else:
                return current_con[2]
        else:
            if current_con[1]:
                return current_con[1]
            elif current_con[2]:
                return current_con[2]
            elif current_con[0]:
                return current_con[0]
            else:
                return current_con[3]
    elif start_pos[0] < current_pos[0] and start_pos[1] < current_pos[1]:
        if current_ori == 1 or current_ori == 2:
            if current_con[3]:
                return current_con[3]
            elif current_con[1]:
                return current_con[1]
            elif current_con[2]:
                return current_con[2]
            else:
                return current_con[0]
        else:
            if current_con[2]:
                return current_con[2]
            elif current_con[0]:
                return current_con[0]
            elif current_con[3]:
                return current_con[3]
            else:
                return current_con[1]
    elif start_pos[0] < current_pos[0] and start_pos[1] == current_pos[1]:
        if current_ori == 2 or current_ori == 0:
            if current_con[1]:
                return current_con[1]
            elif current_con[2]:
                return current_con[2]
            elif current_con[0]:
                return current_con[0]
            else:
                return current_con[3]
        else:
            if current_con[0]:
                return current_con[0]
            elif current_con[3]:
                return current_con[3]
            elif current_con[1]:
                return current_con[1]
            else:
                return current_con[2]
    elif start_pos[0] < current_pos[0] and start_pos[1] > current_pos[1]:
        if current_ori == 2 or current_ori == 0:
            if current_con[1]:
                return current_con[1]
            elif current_con[2]:
                return current_con[2]
            elif current_con[0]:
                return current_con[0]
            else:
                return current_con[3]
        else:
            if current_con[0]:
                return current_con[0]
            elif current_con[3]:
                return current_con[3]
            elif current_con[1]:
                return current_con[1]
            else:
                return current_con[2]
    else:
        return 0
def get_reverse_ori(ori):
    if ori == 0:
        return 1
    elif ori == 1:
        return 0
    elif ori == 2:
        return 3
    else:
        return 2

def copy_random_to(from_path, to_path, file_num):
    all_files = os.listdir(from_path)
    random.shuffle(all_files)
    move_files = all_files[:file_num]
    for single_f_name in move_files:
        shutil.copy(from_path + single_f_name, to_path + single_f_name)

def copy_subfolder_files(from_path, to_path):
    sub_folders = os.listdir(from_path)
    for single_name in sub_folders:
        print(single_name)
        shutil.copy(from_path + single_name + '/bubble_o_boundary_gen0.png', to_path + single_name + 'gen0.png' )


"GT"
def Statistics_GT(data_path):
    all_pkl_paths = [os.path.join(data_path, single_pth) for single_pth in os.listdir(data_path)]
    graph_number = len(all_pkl_paths)
    print(f'graph_num: {graph_number}')

    rooms_num = 0
    liv_connect_num = 0
    non_living_room_num = 0
    living_room_num = 0
    living_room_area = 0


    for single_path in all_pkl_paths:
        with open(single_path, 'rb') as pkl_file:
            [wall_graph, boundary_graph, inter_graph, door_info, room_circles, rooms_info, connects, allG_iteration,
             new_window_mask] = pickle.load(pkl_file)


        rooms_num += len(rooms_info)

        liv_ind = -1
        for i in range(len(rooms_info)):
            if rooms_info[i]['category'] == 0 :
                living_room_num += 1
                liv_ind = i
            else:
                non_living_room_num +=1

        if liv_ind != -1:
            total_area = 0
            for single_room in rooms_info:
                total_area += np.float64(single_room['pixels'])
            living_room_area += rooms_info[liv_ind]['pixels']/total_area

            'liv_connect_num'
            for con in connects:
                if liv_ind in con:
                    liv_connect_num += 1

    print('the all')
    print(rooms_num/graph_number)
    print(liv_connect_num/graph_number)
    print((liv_connect_num/non_living_room_num))
    print(living_room_num/graph_number)
    print(living_room_area/graph_number)

    print(f'liv_connect_num: {liv_connect_num}')
    print(f'non_living_room_num: {non_living_room_num}')


"ours"
def Statistics_ours(data_path):
    all_pkl_paths = [os.path.join(data_path, single_pth) for single_pth in os.listdir(data_path)]
    graph_number = len(all_pkl_paths)
    print(f'graph_num: {graph_number}')

    "统计量"
    rooms_num = 0
    liv_connect_num = 0
    non_living_room_num = 0  #用于计算liv_connect_num/non_living_room_num
    living_room_num = 0
    living_room_area = 0


    for single_path in all_pkl_paths:
        with open(single_path, 'rb') as pkl_file:
            [boundary_graph, rooms_info_array, connects_array, door_info] = pickle.load(pkl_file)

        rooms_info = rooms_info_array[0]
        connects = connects_array[0]
        print(connects)

        'room_num'
        rooms_num += len(rooms_info)

        'non_living_room_num     living_room_num'
        liv_ind = -1
        non_living_room_count = 0
        for i in range(len(rooms_info)):
            if rooms_info[i]['category'] == 0 :
                liv_ind = i
            else:
                non_living_room_count +=1
        if liv_ind!=-1:
            living_room_num +=1
            non_living_room_num += non_living_room_count




        "此时确定living_room号为 liv_ind"
        if liv_ind != -1:
            'living_room_area'
            total_area = 0
            for single_room in rooms_info:
                total_area += np.float64(single_room['pixels'])
            living_room_area += rooms_info[liv_ind]['pixels']/total_area

            'liv_connect_num'
            for con in connects:
                if liv_ind in con:
                    liv_connect_num += 1

    print('the all')
    print(f'graph_num: {graph_number}')
    print(f'living room num: {living_room_num}')

    print(rooms_num/graph_number)
    print(liv_connect_num/graph_number)
    print((liv_connect_num/non_living_room_num))
    print(living_room_num/graph_number)
    print(living_room_area/living_room_num)

    print(f'liv_connect_num: {liv_connect_num}')
    print(f'non_living_room_num: {non_living_room_num}')

def Statistics_DGMG(graph_path):
    with open(graph_path, 'rb') as pkl_graphs:
        [DGMG_graphs] = pickle.load(pkl_graphs)
    rooms_num = 0
    liv_connect_num = 0
    non_living_room_num = 0  # 用于计算liv_connect_num/non_living_room_num
    living_room_num = 0
    living_room_area = 0

    non_living_room_num_when_living = 0



    graph_number = len(DGMG_graphs)
    print(f'图的数量为: {graph_number}')
    for single_graph in DGMG_graphs:
        print(single_graph.nodes)
        print(single_graph.name)
        print(single_graph.nodes[0])
        print(single_graph.edges())

        rooms_num += len(single_graph.nodes)

        liv_ind = -1
        for i in single_graph.nodes:
            if single_graph.nodes[i]!= {}:
                if single_graph.nodes[i]['category'] == 0:
                    living_room_num += 1
                    liv_ind = i
                else:
                    non_living_room_num += 1

        if liv_ind!=-1:
            living_room_area += single_graph.nodes[liv_ind]['area']

            non_living_room_num_when_living += (len(single_graph.nodes)-1)

            for con in single_graph.edges():
                if liv_ind in con:
                    liv_connect_num +=1
    print('the all')
    print(rooms_num / graph_number)
    print(liv_connect_num / living_room_num)
    print((liv_connect_num / non_living_room_num_when_living))
    print(living_room_num / graph_number)
    print(living_room_area / living_room_num)

if __name__ == '__main__':

    "Statistics all data"
    # GT_path = "F:/sjh_study/gs_next_works/RPLAN_data_compact/val/"
    # Statistics_GT(GT_path)
    # ours_boundary_input_path = "F:/sjh_study/Transfloormer/results_pkl/processed_diagrams/"
    # ours_uncondition_input_path = "F:/sjh_study/Transfloormer/results_pkl/unconditionbin_transf_rbf_5.550134_processed/"
    # Statistics_ours(ours_uncondition_input_path)
    # all_graphs_path = "F:/sjh_study/Transfloormer/DGMG_compare/" + "all_graphs.pkl"
    # Statistics_DGMG(all_graphs_path)

    'Statistics our data'
    ablation_vae_path = 'F:/sjh_study/Transfloormer/results_pkl/ablation_vae_no_processed/' #9490个living room，每个布局至少一个


    ablation_2net_path = 'F:/sjh_study/Transfloormer/results_pkl/ablation_2net_no_processed/'  #不一定有布局

    Statistics_ours(ablation_vae_path)










    "比较使用DGMG生成floor plan的最终结果"
    '1. 从origin path到target path，移动所有生成的数据'
    # origin_path = 'F:/sjh_study/Transfloormer/DGMG_compare/DGMG_WallPlan/'    #DGMG的文件数据
    # target_path = 'F:/sjh_study/Transfloormer/DGMG_compare/DGMG_floorplan_compare/DGMG_wallplan/'
    # file_names = os.listdir(origin_path)
    # print(f'f_names: {file_names}')
    # for single_name in file_names:
    #     from_path = origin_path + single_name + '/render_result.png'
    #     if os.path.exists(from_path):
    #         shutil.copyfile(from_path, target_path + single_name + '.png')

    "2. 找ours 和 DGMG生成布局的交集文件"
    # ours_origin_path = 'F:/sjh_study/Transfloormer/results_img/WallPlan/'   #ours + wallplan 生成的 floor plan
    # ours_target_path = 'F:/sjh_study/Transfloormer/DGMG_compare/DGMG_floorplan_compare/ours_wallplan/'
    # file_names = os.listdir(ours_origin_path)
    # print(f'f_names: {file_names}')
    # for single_name in file_names:
    #     from_path = ours_origin_path + single_name + '/render_result_gen_0.png'
    #     if os.path.exists(from_path):
    #         shutil.copyfile(from_path, ours_target_path + single_name + '.png')

    "3. 统一双方的数据，使用相同边界的数据对比"
    # DGMG_result_path = 'F:/sjh_study/Transfloormer/DGMG_compare/DGMG_floorplan_compare/DGMG_wallplan/'
    # OURS_result_path = 'F:/sjh_study/Transfloormer/DGMG_compare/DGMG_floorplan_compare/ours_wallplan/'
    # DGMG_file_names = os.listdir(DGMG_result_path)
    # OURS_file_names = os.listdir(OURS_result_path)
    "删除DGMG中文件"
    # dele_count = 0
    # for single_DGMG in DGMG_file_names:
    #     if single_DGMG not in OURS_file_names:
    #         os.remove(DGMG_result_path + single_DGMG)
    #         dele_count+=1
    #         print(f'删除了第{dele_count}个')

    # dele_count = 0
    # for single_ours in OURS_file_names:
    #     if single_ours not in DGMG_file_names:
    #         os.remove(OURS_result_path + single_ours)
    #         dele_count += 1
    #         print(f'删除了第{dele_count}个')

    "将GT气泡图+wallplan的数据复制，使用相同名"
    # DGMG_result_path = 'F:/sjh_study/Transfloormer/DGMG_compare/DGMG_floorplan_compare/DGMG_wallplan/'
    # gt_wallplan_from_path = 'G:/project_result/data_pool_confirm_12_14/data_pool_vectorization_extraction/vectorization_bubble_ours_8w_extraction/'
    # gt_wallplan_to_path = 'F:/sjh_study/Transfloormer/DGMG_compare/DGMG_floorplan_compare/gt_wallplan/'
    # gt_from_path = 'G:/project_result/data_pool_confirm_12_14/data_pool_vectorization_extraction/vectorization_gt_8w_extraction/'
    #
    # gt_wallplan_file_names = os.listdir(gt_wallplan_from_path)
    # compare_names = os.listdir(DGMG_result_path)
    #
    # for single_gt_wallplan in gt_wallplan_file_names:
    #     if single_gt_wallplan in compare_names:
    #         shutil.copyfile(gt_wallplan_from_path + single_gt_wallplan, gt_wallplan_to_path + single_gt_wallplan)


    "4.统一数据个数"
    # GT_result_path = 'F:/sjh_study/Transfloormer/DGMG_compare/DGMG_floorplan_compare/gt_wallplan/'
    # OURS_result_path = 'F:/sjh_study/Transfloormer/DGMG_compare/DGMG_floorplan_compare/ours_wallplan/'
    # DGMG_result_path = 'F:/sjh_study/Transfloormer/DGMG_compare/DGMG_floorplan_compare/DGMG_wallplan/'
    #
    # GT_file_names = os.listdir(GT_result_path)
    # DGMG_file_names = os.listdir(DGMG_result_path)
    # OURS_file_names = os.listdir(OURS_result_path)
    #
    # for single_DGMG in DGMG_file_names:
    #     if single_DGMG not in GT_file_names:
    #         os.remove(DGMG_result_path + single_DGMG)


    # for single_ours in OURS_file_names:
    #     if single_ours not in GT_file_names:
    #         os.remove(OURS_result_path + single_ours)

    "结束可以比较FID了"













