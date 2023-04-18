# Mine-HRL

## Requirements
`pip install -r requirements.txt`

## Test wrapped environment with stable baselines
`python main_sb3.py`

## Tasks
- Level 1 - Get wood: randomly generate few woods in the environment, just need to get one of them

- Level 2 - Build plank: get a wood, and use the wood in the workbench to get the plank; Build bridge: get a wood and a iron, and use the workbench


## Reference
Environment:
[psketch](https://github.com/jacobandreas/psketch)

[Program synthesis guided RL](https://github.com/yycdavid/program-synthesis-guided-RL)

[ProTo](https://github.com/sjtuytc/Neurips21-ProTo-Program-guided-Transformers-for-Program-guided-Tasks)

Please implement those agents in trainers (one folder for each), and add an entry in `main_HRL.py`:
- Task A+B:
    [Option-critic](https://github.com/lweitkamp/option-critic-pytorch) and the related work

    [DSC](https://github.com/deep-skill-chaining/deep-skill-chaining)

    [FuN](https://github.com/lweitkamp/feudalnets-pytorch)

- Task B:
    [h-DQN](https://github.com/skumar9876/Hierarchical-DQN)

    [HIRO](https://github.com/watakandai/hiro_pytorch)

    [HAC](https://github.com/andrew-j-levy/Hierarchical-Actor-Critc-HAC-)


# TODO
- Implementing agents above (manage the hyperparameters in a yaml for each algorithm)
- Plotting (curve for steps to reach the goals, successful rate, ...)
- Systematic logging
- Different Levels (map size, task complexity (horizon required), OOD, stochasticity (both levels))
