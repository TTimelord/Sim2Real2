agent_cfg = dict(
    type="CEM",
    cem_cfg=dict(
        n_iter=5,
        population=200,
        elite=10,
        lr=1.0,
        temperature=1.0,
        # use_trunc_normal=True,
        use_softmax=False,
    ),
    scheduler_config=dict(type="FixedScheduler"),
    horizon=6,
    add_actions=True,
    action_horizon=1,
)

rollout_cfg = dict(
    type="Rollout",
    num_procs=20,
)

log_level = "INFO"

eval_cfg = dict(
    type="Evaluation",
    num_procs=1,
    use_hidden_state=True,
    start_state=None,
    save_traj=True,
    save_video=True,
    use_log=True,
    save_info=False,
    log_every_step=True,
)
