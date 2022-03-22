import numpy as np
import math, os, sys
sys.path.append('.')
import configs as cfg
import copy

def get_nearby_pos_cand(target_obj, cand_objects, all_objects, nearby_names, is_stack=False):
    nearby_pos_cand = list()
    # find straightly putted objects, and nearby position around the objects.
    if len(find_obj_candidate(target_obj, cand_objects)) > 0:
        for obj_cand in find_obj_candidate(target_obj, cand_objects):
            tmp_pos_cand, tmp_move_name = find_obj_nearby_location(obj_cand, nearby_names)
            for idx, p in enumerate(tmp_pos_cand):
                search_objs = [obj for obj in all_objects if obj != obj_cand]
                if check_emptyness(p, search_objs, is_stack) and check_bottom(p, [o for o in all_objects if o!= target_obj]):
                    nearby_pos_cand.append([obj_cand, p, tmp_move_name[idx]])
    return nearby_pos_cand
    


def check_upper_blocks(curr_obj, objects):
    curr_coord = np.asarray(curr_obj['3d_coords'])

    for obj in objects:
        tmp_coord = np.asarray(obj['3d_coords'])
        if abs(curr_coord[0]-tmp_coord[0]) < cfg.eps and abs(curr_coord[1]-tmp_coord[1]) < cfg.eps:
            if tmp_coord[2] - curr_coord[2] > cfg.block_size/2:
                return True

    return False

def check_bottom(position, objects):
    if abs(position[2]-cfg.block_size) < cfg.eps:
        return True
        
    for obj in objects:
        tmp_coord = np.asarray(obj['3d_coords'])
        if abs(tmp_coord[0]-position[0]) < cfg.eps and abs(tmp_coord[1]-position[1]) < cfg.eps:
            if abs(position[2]-tmp_coord[2]-cfg.block_size*2) < cfg.eps:
                return True
    return False

def find_obj_nearby_location(obj, nearby_names):
    candidates = list()
    names = list()
    for n, v in cfg.strides.items():
        if n in nearby_names:
            tmp_pos = copy.deepcopy(obj['3d_coords'])
            for i in range(3):
                tmp_pos[i] += v[i]
            candidates.append(tmp_pos)
            names.append(n)
    return candidates, names


# find object that is straightly placed, except for the target object
def find_obj_candidate(target_obj, objects):
    search_objs = [o for o in objects if o != target_obj]#[objects[i] for i in range(len(objects)) if i != target_obj_idx]
    candidates = list()
    for obj in search_objs:
        for rad in [0.0, math.pi/2, math.pi, math.pi*1.5, math.pi*2]:
            if abs(obj['rotation'] - rad) < np.deg2rad(cfg.rot_thres):
                candidates.append(obj)
                break

    return candidates


def find_empty_location(target_obj, objects):
    search_objs = [o for o in objects if o != target_obj] #[objects[i] for i in range(len(objects)) if i != target_obj]
    candidates = list()
    for name, value in cfg.positions.items():
        if check_emptyness(value, search_objs):
            candidates.append(name)

    return candidates


def check_emptyness(position, objects, is_stack=False):
    """
    position : 3 dimensional numpy vector
    """
    for obj in objects:
        obj_coord = np.asarray(obj['3d_coords'])
        dist = np.linalg.norm(obj_coord[:2]-position[:2])
        if obj_coord[-1] == position[-1]:
            if is_stack:
                if dist < cfg.block_size*2: # note that block size is fixed to 0.5
                    return False
            else:
                if dist < cfg.eps + cfg.block_size*2: # note that block size is fixed to 0.5
                    return False
                
            
        if abs(position[0]) > cfg.pos_bound or abs(position[1]) > cfg.pos_bound:
            return False

    return True
        