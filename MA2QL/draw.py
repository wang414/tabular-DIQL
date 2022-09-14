from tensorboard.backend.event_processing import event_accumulator
import os
from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np
import argparse

critical_value = 1.96

frame_list = ['maac','coma','qmix']
frame_alg_dict = {}
for frame in frame_list:
    if frame != 'maac':
        frame_alg_dict[frame] = [frame + '_div_soft',frame,frame + '_div_sac']
    else:
        frame_alg_dict[frame] = [frame + '_div_soft', frame]
frame_alg_dict['dapo'] = ['coma_div_soft', 'coma',  'dapo','maddpg']
frame_alg_dict['aga'] = ['aga']


parser = argparse.ArgumentParser()
parser.add_argument("--sample_num",
                        default=1, type=int,
                        help="Batch size for training")
parser.add_argument("--env_id",
                        default='3m', type=str,)
parser.add_argument("--frame",
                        default='coma', type=str,)
parser.add_argument("--model_name", default='easy_v2_c0')
parser.add_argument("--extra_alg",default=None,type=str)
parser.add_argument("--mode",default='step',type=str)
parser.add_argument("--run_num",default=None,type=str)
parser.add_argument("--multi_draw",default=False,type=bool)
parser.add_argument("--single_run", default=False, type=bool)
parser.add_argument("--width",type = float,default=18.0)
parser.add_argument("--height",type = float,default=10.0)
parser.add_argument("--wspace",type = float,default=None)
parser.add_argument("--hspace",type = float,default=0.3)
parser.add_argument("--clear_plot",type = bool,default=False)
parser.add_argument("--ex_name_list", type=str)
config = parser.parse_args()
if config.run_num is not None:
    config.run_num = eval(config.run_num)
model_name = config.model_name
single_run = config.single_run
ex_name_list = config.ex_name_list.split(',')

sample_num = config.sample_num




frame = config.frame

alg_to_color = {}
alg_to_color['maac_div_soft'] = 'r'
alg_to_color['maac'] = 'b'
alg_to_color['coma_div_soft'] = 'r'
alg_to_color['coma'] = 'b'
alg_to_color['coma_div_sac'] = 'g'
alg_to_color['qmix_div_soft'] = 'r'
alg_to_color['qmix'] = 'b'
alg_to_color['qmix_div_sac'] = 'g'
alg_to_color['dapo'] = 'y'
alg_to_color['maddpg'] = 'm'
alg_draw_name = {}
alg_from_list = ['dapo','maddpg','maac','maac_div_soft','maac-share','maac-share_div_soft','coma', 'coma_div_soft','coma_div_sac','qmix', 'qmix_div_soft','qmix_div_sac']
alg_to_list = ['DAPO','MADDPG','MAAC','MAAC+DMAC','MAAC','MAAC+DMAC','COMA', 'COMA+DMAC','COMA+ENT','QMIX', 'QMIX+DMAC','QMIX+ENT']


for a,b in zip(alg_from_list,alg_to_list):
    alg_draw_name[a] = b

if frame == 'aga':
    env_id_list = [config.env_id]
else:
    env_id_list = ['3m','2s3z','3s5z','8m','1c3s5z']



def get_data_from_tf_event(model_dir,sample_num):
    exist_run_nums = [int(str(folder.name).split('run')[1]) for folder in
                      model_dir.iterdir() if
                      str(folder.name).startswith('run')]
    max_run_num = max(exist_run_nums)
    # print('env_id {} alg_name {}'.format(env_id, alg_name))
    # print('exist_run_nums = {}'.format(exist_run_nums))
    # print('max_run_num = {}'.format(max_run_num))
    if check_end:
	    while True:
	        curr_run = 'run%i' % max_run_num
	        run_dir = model_dir / curr_run
	        log_dir = run_dir / 'logs'

	        print(os.listdir(log_dir))
	        event_name = os.listdir(log_dir)[0]
	        if (len(os.listdir(log_dir)) < 2):
	            max_run_num -= 1
	        else:
	            break
    ret_steps = {}
    ret_performance = {}
    for i in range(sample_num):
        if i >= max_run_num:
            break
        run_num = max_run_num - i
        # print('loading data for:run_num {}'.format(run_num))
        curr_run = 'run%i' % run_num
        run_dir = model_dir / curr_run
        log_dir = run_dir / 'logs'
        # event_name = os.listdir(log_dir)[0]
        for name in os.listdir(log_dir):
            if name.startswith('events'):
                event_name = name
                break

        ea = event_accumulator.EventAccumulator(str(log_dir / event_name))
        ea.Reload()
        # print(ea)
        # print('prev all_key = {}'.format(ea.scalars.Keys()))
        all_key = []
        for key in ea.scalars.Keys():
            if key.startswith('agent') and not key.startswith('agent0'):
                continue
            all_key.append(key)
        # print('after all_key = {}'.format(all_key))
        for key in all_key:
            print('load data for ::\nmodel_dir = {}\nsample_num = {}\nkey = {}'.format(model_dir,sample_num,key))
            performance = ea.scalars.Items(key)
            # print(performance)

            run_steps = [i.step for i in performance]
            run_val = [i.value for i in performance]

            if key not in ret_steps:
                ret_steps[key] = run_steps
            if key not in ret_performance:
                ret_performance[key] = []
            ret_performance[key].append(run_val)
    return ret_performance,ret_steps,all_key


def get_data_from_npy(model_dir,sample_num,key_list,run_num = None):
    if run_num is None:
        exist_run_nums = [int(str(folder.name).split('run')[1]) for folder in
                          model_dir.iterdir() if
                          str(folder.name).startswith('run')]
        if len(exist_run_nums) == 0:
            return None, None
        max_run_num = max(exist_run_nums)
        sample_list = max_run_num - np.array(range(sample_num))
    else:
        sample_list = run_num
    # print('env_id {} alg_name {}'.format(env_id, alg_name))
    # print('exist_run_nums = {}'.format(exist_run_nums))
    # print('max_run_num = {}'.format(max_run_num))

    ret_steps = {}
    ret_performance = {}
    min_length = 1000000000
    for key in key_list:
        min_length = 1000000000
        for i in sample_list:
            if i <= 0:
                break
            run_num = i
            # print('loading data for:run_num {}'.format(run_num))
            file_path = model_dir / '{}_{}.npy'.format(key,run_num)
            # print('file_path = {}'.format(file_path))
            if not Path(file_path).exists():
                # print('no file skip!')
                continue
            run_val = np.load(file_path)
            min_length = min(min_length, len(run_val) )
            # print('{}: shape = {}'.format(key, np.shape(run_val)))
            if key not in ret_performance:
                ret_performance[key] = []
            ret_performance[key].append(run_val)
            # print('ret_performance.key = {}'.format(ret_performance.keys()))
        # print('min_length for key {} is {}'.format(key,))
        for i in range(len(ret_performance[key])):
            ret_performance[key][i] = ret_performance[key][i][0:min_length]

    for key in ret_performance:
        ret_performance[key] = np.array(ret_performance[key])
    return ret_performance,ret_steps


def multi_plot_draw(config,test_para='dim_test', line_alpha=0.5):
    sample_num = config.sample_num
    mode = config.mode
    if not Path('plot').exists():
        os.mkdir('plot')
    origin_name = 'plot' + '/' + test_para
    if not Path(origin_name).exists():
        os.mkdir(origin_name)


    performance_all = {}
    steps_all = {}
    key_list = ['episode_rewards','win_rates','episode_count','t_env']
    draw_key_list = ['episode_rewards', 'win_rates']
    para = '64x32'
    for count,env_id in enumerate(env_id_list):
        performance_all[env_id] = {}
        steps_all[env_id] = {}
        for f_id,frame in enumerate(frame_list):
            performance_all[env_id][frame] = {}
            steps_all[env_id][frame] = {}
            alg_list = frame_alg_dict[frame]
            for a_id, alg in enumerate(alg_list):
                performance_all[env_id][frame][alg] = {}
                steps_all[env_id][frame][alg] = {}
                model_dir = Path('result') / alg / env_id / para
                if not model_dir.exists():
                    continue
                if config.run_num is not None:
                    p_ret,s_ret = get_data_from_npy(model_dir, sample_num, key_list,config.run_num[count][f_id][a_id])
                else:
                    p_ret,s_ret = get_data_from_npy(model_dir, sample_num,key_list)
                if p_ret is None:
                    continue
                performance_all[env_id][frame][alg][para], steps_all[env_id][frame][alg][para] = p_ret,s_ret
                print('path = {}'.format(model_dir))

    for key in draw_key_list:
        myfontsize = 8
        plt.rcParams.update({"font.size": myfontsize})
        # plt.rcParams['figure.figsize'] = (4.0, 8.0)
        # plt.rcParams['savefig.dpi'] = 300  # 图片像素
        # plt.rcParams['figure.dpi'] = 300  # 分辨率
        plt.figure(figsize=(config.width, config.height), dpi=300)

        fig = plt.figure(1)
        fig.subplots_adjust(wspace=config.wspace, hspace=config.hspace)

        name_key = key
        for count, env_id in enumerate(env_id_list):
            for f_id, frame in enumerate(frame_list):
                plot_num = len(frame_list) * count + f_id + 1
                print('plot_num = {}'.format(plot_num))
                ax = plt.subplot(len(env_id_list), len(frame_list), plot_num)
                for alg in performance_all[env_id][frame]:
                    if para not in performance_all[env_id][frame][alg]:
                        continue
                    if key not in performance_all[env_id][frame][alg][para]:
                        continue

                    performance_all[env_id][frame][alg][para][key] = np.array(performance_all[env_id][frame][alg][para][key])
                    mean_val = performance_all[env_id][frame][alg][para][key].mean(axis=0)
                    std_val = performance_all[env_id][frame][alg][para][key].std(axis=0)
                    bound_val = critical_value * std_val / sample_num
                    if mode == 'step':
                        # run_step = performance_all[alg][para]['t_env'].mean(axis = 0)
                        run_step = np.array(range(len(mean_val))) * 5000
                        if plot_num >= len(frame_list) * (len(env_id_list) - 1) + 1:
                            plt.xlabel('step', fontsize=myfontsize)
                    elif mode == 'epoch':
                        if 'episode_count' not in performance_all[env_id][frame][alg][para]:
                            run_step = range(len(mean_val))

                            if plot_num >= len(frame_list) * (len(env_id_list) - 1) + 1:
                                plt.xlabel('episode*100', fontsize=myfontsize)
                        else:
                            run_step = performance_all[env_id][frame][alg][para]['episode_count'].mean(axis=0)
                            if plot_num >= len(frame_list) * (len(env_id_list) - 1) + 1:
                                plt.xlabel('episode', fontsize=myfontsize)
                    plt.plot(run_step, mean_val, label='{}'.format(alg_draw_name[alg]), alpha=line_alpha,
                             color=alg_to_color[alg])
                    # plt.legend(loc=2, bbox_to_anchor=(1.05, 1.0), borderaxespad=0)
                    if plot_num <= len(frame_list):
                        plt.legend(loc='lower right', ncol=3, frameon=False, borderpad=0, framealpha=0.2)
                    plt.fill_between(run_step, mean_val - bound_val, mean_val + bound_val, alpha=0.3,
                                     color=alg_to_color[alg])
                plt.title(env_id,fontsize=myfontsize)
                plt.xticks(fontsize=myfontsize)
                plt.yticks(fontsize=myfontsize)
                if plot_num % len(frame_list) == 1:
                    plt.ylabel(key, fontsize=myfontsize)

        plt.savefig(
            str(origin_name) + '/' + 'SC2_ALL_' + name_key + '_{}_result.pdf'.format(mode),
            dpi=300, bbox_inches='tight',pad_inches = 0)
        plt.clf()


def draw_diff(config,test_para='dim_test', line_alpha=0.5):
    sample_num = config.sample_num
    mode = config.mode
    if not Path('plot').exists():
        os.mkdir('plot')
    origin_name = 'plot' + '/' + test_para
    if not Path(origin_name).exists():
        os.mkdir(origin_name)
    frame = config.frame
    alg_list = frame_alg_dict[frame]
    if config.env_id == 'random_matrix_game_3' and frame == 'coma':
        alg_list = ['coma','coma_div_soft']
    print('alg_list = {}'.format(alg_list))
    if config.extra_alg is not None:
        alg_list.append(config.extra_alg)
    if config.env_id == 'random_matrix_game_3':
        key_list = ['traverse', 'episode_count', 't_env']
        draw_key_list = ['traverse']
    else:
        key_list = ['episode_rewards','win_rates','episode_count','t_env']
        draw_key_list = ['episode_rewards', 'win_rates']
    env_id_list = [config.env_id]
    para_list = ['64x32']
    for count, env_id in enumerate(env_id_list):

        performance_all = {}
        steps_all = {}

        for a_id,alg in enumerate(alg_list):
            performance_all[alg] = {}
            steps_all[alg] = {}
            for para in para_list:
                model_dir = Path('result') / alg / env_id / para
                print('model_dir = {}'.format(model_dir))
                if not model_dir.exists():
                    continue
                if config.run_num is not None:
                    p_ret, s_ret = get_data_from_npy(model_dir, sample_num, key_list,config.run_num[a_id])
                else:
                    p_ret,s_ret = get_data_from_npy(model_dir, sample_num,key_list)
                if p_ret is None:
                    continue
                performance_all[alg][para], steps_all[alg][para] = p_ret,s_ret
                print('path = {}'.format(model_dir))

        for key in draw_key_list:
            plt.figure(figsize=(6.0,4.0))
            name_key = key
            for alg in alg_list:
                for para in para_list:
                    if para not in performance_all[alg]:
                        continue
                    if key not in performance_all[alg][para]:
                        continue
                    performance_all[alg][para][key] = np.array(performance_all[alg][para][key])
                    print('performance_shape = {}'.format(np.shape(performance_all[alg][para][key])))
                    mean_val = performance_all[alg][para][key].mean(axis=0)
                    std_val = performance_all[alg][para][key].std(axis=0)
                    bound_val = critical_value * std_val / sample_num
                    if mode == 'step':
                        # run_step = performance_all[alg][para]['t_env'].mean(axis = 0)
                        if config.env_id == 'random_matrix_game_3':
                            run_step = np.array(range(len(mean_val))) * 120
                        else:
                            run_step = np.array(range(len(mean_val))) * 5000
                        plt.xlabel('step')
                    elif mode == 'epoch':
                        if 'episode_count' not in performance_all[alg][para]:
                            run_step = range(len(mean_val))
                            plt.xlabel('episode*100')
                        else:
                            run_step = performance_all[alg][para]['episode_count'].mean(axis = 0)
                            plt.xlabel('episode')


                    if name_key == 'traverse':
                        name_key = 'cover_rate'
                    if env_id == 'random_matrix_game_3':
                        env_id_to_draw = 'random_matrix_game'
                    else:
                        env_id_to_draw = env_id
                    plt.title(env_id_to_draw)
                    plt.ylabel(name_key)
                    plt.plot(run_step, mean_val, label='{}'.format(alg_draw_name[alg]), alpha=line_alpha,color=alg_to_color[alg])
                    plt.legend(loc='center right', ncol=1, frameon=False, borderpad=0, framealpha=0.2)

                    plt.fill_between(run_step, mean_val - bound_val, mean_val + bound_val, alpha=0.3,color=alg_to_color[alg])

            plt.savefig(
                str(origin_name) + '/' + frame + '_' + env_id + '_' + name_key + '_{}_result.pdf'.format(mode),
                dpi=300, bbox_inches='tight',pad_inches = 0)
            plt.clf()

def single_alg_draw():
    mode = config.mode
    dir_name = frame + '_' + model_name
    plot_dir_name = 'plot/'+dir_name
    alg_name_list = frame_alg_dict[frame]

    if not Path('plot').exists():
        os.mkdir('plot')
    if not Path(plot_dir_name).exists():
        os.mkdir(plot_dir_name)
    key_list = ['episode_rewards', 'win_rates', 'episode_count', 't_env']
    for count, env_id in enumerate(env_id_list):

        steps = {}
        performance_val = {}
        all_key = {}
        sub_plot = plt
        for a_id, alg_name in enumerate(alg_name_list):
            performance_val[alg_name], steps[alg_name], all_key[alg_name] = {}, {} , {}
            for ex_id,ex_name in enumerate(ex_name_list):

                model_dir = Path('./result')/alg_name  /env_id / ex_name
                print('model_dir = {}'.format(model_dir))
                performance_val[alg_name][ex_name], steps[alg_name][ex_name] = get_data_from_npy(model_dir, sample_num,key_list)

        # print('all_key.key = {}'.format(all_key.keys()))
        # print('all_key[{}].key = {}'.format(alg_name_list[0],all_key[alg_name_list[0]].keys()))
        all_key_eg = key_list
        # print('all_key_eg = {}'.format(all_key_eg))
        if single_run:
            print('fuck')
            for sample_idx in range(sample_num):
                single_dir_name = dir_name + '/run{}'.format(sample_idx + 1)
                if not Path(single_dir_name).exists():
                    os.mkdir(single_dir_name)
                for key in all_key_eg:
                    # plt.figure(figsize=(6.0, 4.0))
                    print('draw_plot for {} in run {}'.format(key, sample_idx + 1))
                    if key.startswith('agent0'):
                        name_key = 'rewards'
                    else:
                        name_key = key
                    for alg_name in performance_val:
                        for ex_name in performance_val[alg_name]:
                            if key not in performance_val[alg_name][ex_name].keys() or key != 'eval_reward':
                                continue
                            performance_val[alg_name][ex_name][key] = np.array(performance_val[alg_name][ex_name][key])
                            mean_val = performance_val[alg_name][ex_name][key][sample_idx]
                            std_val = np.zeros_like(performance_val[alg_name][ex_name][key][sample_idx])
                            bound_val = critical_value * std_val / sample_num
                            # print('bound_val = {}'.format(bound_val))
                            # run_step = steps[alg_name][ex_name][key]

                            if mode == 'step':
                                # run_step = performance_all[alg][para]['t_env'].mean(axis = 0)
                                run_step = np.array(range(len(mean_val))) * 5000
                            elif mode == 'epoch':
                                if 'episode_count' not in performance_val[alg_name][ex_name].keys():
                                    run_step = range(len(mean_val))

                                else:
                                    run_step = performance_val[alg_name][ex_name]['episode_count'].mean(axis=0)

                            length = len(mean_val)
                            idx = range(0, length, 10)
                            mean_val, std_val, bound_val, run_step = np.array(mean_val), np.array(std_val), np.array(
                                bound_val), np.array(run_step)
                            if config.clear_plot:
                                mean_val, std_val, bound_val, run_step = mean_val[idx], std_val[idx], bound_val[idx], \
                                                                         run_step[
                                                                             idx]

                            sub_plot.title(env_id)
                            sub_plot.xlabel('episode')
                            sub_plot.ylabel(name_key)
                            sub_plot.plot(run_step, mean_val, label=ex_name, alpha=0.5)
                            sub_plot.fill_between(run_step, mean_val - bound_val, mean_val + bound_val, alpha=0.3)
                            # plt.legend(loc='lower right')
                            plt.legend(loc=2, bbox_to_anchor=(1.05, 1.0), borderaxespad=0,prop = {'size':2})

                        print('name_key = {}'.format(name_key))
                        # print('run_step = {}'.format(run_step))

                        plt.savefig(
                            single_dir_name + '/' + frame +'_{}_{}'.format(model_name,alg_name) + '_' + env_id + '_' + name_key + '_result.png',
                            dpi=300)
                        plt.clf()
        else:
            for key in all_key_eg:
                # plt.figure(figsize=(6.0, 4.0))
                print('draw_plot for {}'.format(key))
                if key.startswith('agent0'):
                    name_key = 'rewards'
                else:
                    name_key = key
                for alg_name in performance_val:
                    for ex_name in performance_val[alg_name]:
                        print('key = {}, performance_val[{}][{}].keys = {}'.format(key,alg_name,ex_name,performance_val[alg_name][ex_name].keys()))
                        if key not in performance_val[alg_name][ex_name].keys():
                            continue
                        performance_val[alg_name][ex_name][key] = np.array(performance_val[alg_name][ex_name][key])
                        mean_val = performance_val[alg_name][ex_name][key].mean(axis=0)
                        std_val = performance_val[alg_name][ex_name][key].std(axis=0)
                        bound_val = critical_value * std_val / sample_num
                        # print('bound_val = {}'.format(bound_val))
                        if mode == 'step':
                            # run_step = performance_all[alg][para]['t_env'].mean(axis = 0)
                            run_step = np.array(range(len(mean_val))) * 5000
                        elif mode == 'epoch':
                            if 'episode_count' not in performance_val[alg_name][ex_name].keys():
                                run_step = range(len(mean_val))

                            else:
                                run_step = performance_val[alg_name][ex_name]['episode_count'].mean(axis=0)

                        length = len(mean_val)
                        idx = range(0, length, 10)
                        mean_val, std_val, bound_val, run_step = np.array(mean_val), np.array(std_val), np.array(
                            bound_val), np.array(run_step)
                        if config.clear_plot:
                            mean_val, std_val, bound_val, run_step = mean_val[idx], std_val[idx], bound_val[idx], run_step[
                                idx]

                        sub_plot.title(env_id)
                        sub_plot.xlabel('episode')
                        sub_plot.ylabel(name_key)
                        sub_plot.plot(run_step, mean_val, label=ex_name, alpha=0.5)
                        sub_plot.fill_between(run_step, mean_val - bound_val, mean_val + bound_val, alpha=0.3)
                        plt.legend(loc='lower right',prop = {'size':4})
                        # plt.legend(loc=2, bbox_to_anchor=(1.05, 1.0), borderaxespad=0,prop = {'size':2})

                    print('name_key = {}'.format(name_key))
                    # print('run_step = {}'.format(run_step))

                    plt.savefig(plot_dir_name + '/' + frame +'_{}_{}'.format(model_name,alg_name) + env_id + '_' + name_key + '_result.png',
                                dpi=300)
                    plt.clf()



# if config.multi_draw:
#     multi_plot_draw(config = config)
# else:
#     draw_diff(config = config)

single_alg_draw()