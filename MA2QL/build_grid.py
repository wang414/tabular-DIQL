import pickle
from copy import deepcopy
key_list = ['batch_size','train_steps','reset_epsilon_min','stage_lr']
para_dict = {'batch_size':[32,128,512],'train_steps':[5,10,15],'reset_epsilon_min':[0.05,0.02,0.0],'stage_lr':[True,False]}
grids = []
def build_grids(curr_depth,grid,max_depth=4,):
    if curr_depth == max_depth:
        grids.append(grid)
        return
    key = key_list[curr_depth]
    para_list = para_dict[key]
    for para in para_list:
        new_grid = deepcopy(grid)
        new_grid[key] = para
        build_grids(curr_depth + 1, new_grid,max_depth)
    return

build_grids(0,{})
print(grids)
print(len(grids))
with open('grids_para.pkl','wb') as f:
    pickle.dump(grids,f)
with open('grids_key.pkl','wb') as f:
    pickle.dump(key_list,f)
print('grids[26] = {}'.format(grids[26]))