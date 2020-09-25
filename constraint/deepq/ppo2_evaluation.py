from baselines.common.policies import build_policy


def main(network, env, **network_kwargs):
    policy = build_policy(env, network, **network_kwargs)

    # Get state_space and action_space
    ob_space = env.observation_space
    ac_space = env.action_space
    nenvs = env.num_envs

    # Instantiate the model object (that creates act_model and train_model)
    if model_fn is None:
        from baselines.ppo2.model import Model
        model_fn = Model

    model = model_fn(policy=policy,
                     ob_space=ob_space,
                     ac_space=ac_space,
                     nbatch_act=nenvs,
                     nbatch_train=nbatch_train,
                     nsteps=nsteps,
                     ent_coef=ent_coef,
                     vf_coef=vf_coef,
                     max_grad_norm=max_grad_norm,
                     comm=comm,
                     mpi_rank_weight=mpi_rank_weight)

    if load_path is not None:
        model.load(load_path)