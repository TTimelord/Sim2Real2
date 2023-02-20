# Data generation
1. change urdf dir
2. change scale to the real scale, then check the scale_rand_ratio:
   1. laptop*1/3.2
   2. drawer 0.333
   3. faucet 0.2
3. change initial joint state
4. check friction
5. check traj length
   1. laptop
   2. faucet 0.06 (pushing-left)
6. check camera distance

# Train Critic
1. change data dir in .sh
2. change offline data number

# testing
1. change urdf dir
2. change initial state
3. change scale