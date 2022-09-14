with open('grid_search_code.txt','w') as f:
    for i in range(18):
        f.write('nohup python grid_search.py --alg aga --aga_tag True   --idv_para True --map 3m  --train_steps 10 --single_exp True --stop_other_exp True  --epsilon_reset True --optim_reset True --epsilon_reset_factor 1 --t_max 5005000 --cuda True --grid_search_idx [{},{},{}] > case2.out  2>&1   &\n'.format(3*i,3*i + 1,3*i+2))