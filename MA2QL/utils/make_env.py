"""
Code for creating a multiagent environment with one of the scenarios listed
in ./scenarios/.
Can be called by using, for example:
    env = make_env('simple_speaker_listener')
After producing the env object, can be used similarly to an OpenAI gym
environment.

A policy using this environment must output actions in the form of a list
for all agents. Each element of the list should be a numpy array,
of size (env.world.dim_p + env.world.dim_c, 1). Physical actions precede
communication actions in this array. See environment.py for more details.
"""

def make_env(scenario_name, benchmark=False, discrete_action=False,collision_penal = 0,vision = 1):
    '''
    Creates a MultiAgentEnv object as env. This can be used similar to a gym
    environment by calling env.reset() and env.step().
    Use env.render() to view the environment on the screen.

    Input:
        scenario_name   :   name of the scenario from ./scenarios/ to be Returns
                            (without the .py extension)
        benchmark       :   whether you want to produce benchmarking data
                            (usually only done during evaluation)

    Some useful env properties (see environment.py):
        .observation_space  :   Returns the observation space for each agent
        .action_space       :   Returns the action space for each agent
        .n                  :   Returns the number of Agents
    '''
    from multiagent.environment import MultiAgentEnv
    import multiagent.scenarios as old_scenarios
    import envs.mpe_scenarios as new_scenarios

    # load scenario from script
    try:
        scenario = old_scenarios.load(scenario_name + ".py").Scenario()
    except:
        scenario = new_scenarios.load(scenario_name + ".py").Scenario()
    # create world
    world = scenario.make_world(collision_penal = collision_penal,vision = vision)
    # create multiagent environment
    if hasattr(scenario, 'post_step'):
        post_step = scenario.post_step
    else:
        post_step = None
    if benchmark:        
        env = MultiAgentEnv(world, reset_callback=scenario.reset_world,
                            reward_callback=scenario.reward,
                            observation_callback=scenario.observation,
                            post_step_callback=post_step,
                            info_callback=scenario.benchmark_data,
                            done_callback=scenario.done,
                            discrete_action=discrete_action)
    else:
        env = MultiAgentEnv(world, reset_callback=scenario.reset_world,
                            reward_callback=scenario.reward,
                            observation_callback=scenario.observation,
                            post_step_callback=post_step,
                            done_callback=scenario.done,
                            discrete_action=discrete_action)
    return env

from smac.env import StarCraft2Env
def make_sc2_env(args,rank):
    env =  StarCraft2Env(map_name=args.map,
                            step_mul=args.step_mul,
                            difficulty=args.difficulty,
                            game_version=args.game_version,
                            replay_dir=args.replay_dir,
                         seed = args.seed + 1000 * rank)
    return env
