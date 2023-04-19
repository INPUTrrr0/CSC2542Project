# Mine-HRL

## Requirements
`pip install -r requirements.txt`

## Test wrapped environment with stable standard RL baselines
`python main_sb3.py`

- If you are not familiar with the environment, run `main_sb3.py` in debug mode, and the environment states will be verbosely printed, where current location is printed as -1

## Test wrapped environment with HRL baselines
`python main_HRL.py`

## Tasks
Specify task configurations in `main_xxx.py`, first line of `configure()`
- Level 1 - Get wood (1 option): randomly generate few woods in the environment, just need to get one of them;

- Level 2 - Build plank (2 options): get a wood, and use the wood in the workbench to get the plank; Build bridge (3 options): get a wood and a iron, and use the workbench; 

- Level 3 - Build bed (4 options): get a wood and a grass, use workbench_0 to convert the wood into a plank, then use workbench_1 to convert the plank and grass into a bed; 

- Level 4 - Get gem (6 options): get wood, iron, use workbench_0 to convert the wood into a stick, then use workbench_1 to convert the stick and iron into a axe, use axe to break stone, and get gem;

- Variants: larger/smaller map size, more/less resource, OOD, stochasticity

Noted that, "option" here means how many human instructions given (ideally it may learn those options). Of course, human can also regard all "get" as an option.


Craft your own task:
- add an config in `./experiments/`

- add an recipe in `./resources/` (refer to `recipes_all.yaml`)

- Specify task configurations in `main_xxx.py`

## Reference
Environment:
[psketch](https://github.com/jacobandreas/psketch)

[Program synthesis guided RL](https://github.com/yycdavid/program-synthesis-guided-RL)

[ProTo](https://github.com/sjtuytc/Neurips21-ProTo-Program-guided-Transformers-for-Program-guided-Tasks)

Please implement those agents in trainers (one folder for each), and add an entry in `main_HRL.py`:
- Task A+B:
    [Option-critic](https://github.com/lweitkamp/option-critic-pytorch) and the related work ([Proximal Policy OC](https://github.com/mklissa/PPOC), [Interest OC](https://github.com/kkhetarpal/ioc))

    [DSC](https://github.com/deep-skill-chaining/deep-skill-chaining)

    [FuN](https://github.com/lweitkamp/feudalnets-pytorch)

- Task B:
    [h-DQN](https://github.com/skumar9876/Hierarchical-DQN)

    [HIRO](https://github.com/watakandai/hiro_pytorch)

    [HAC](https://github.com/andrew-j-levy/Hierarchical-Actor-Critc-HAC-)


# TODO
- Implementing agents above (manage the hyperparameters in a yaml for each algorithm)
- Give more reward (>1) if it uses less time steps
- Eval mode
- Plotting (curve for steps to reach the goals, successful rate, ...)
- Systematic logging
- Different Levels (map size, task complexity (horizon required), OOD, stochasticity (both levels))
- Better state repr (currently we are using simple tricks, and bag (inventory) state is redundant)
