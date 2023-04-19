from environment.cookbook import Cookbook
from utils import array

import curses
import logging
import numpy as np
from skimage.measure import block_reduce
import time

# WIDTH = 10
# HEIGHT = 10

# WINDOW_WIDTH = 5
# WINDOW_HEIGHT = 5

# N_WORKSHOPS = 3

DOWN = 0
UP = 1
LEFT = 2
RIGHT = 3
USE = 4
N_ACTIONS = USE + 1


def random_free(grid, random, width, height):
    """
    Find some random and free (no object) place in the map
    """
    pos = None
    while pos is None:
        (x, y) = (random.randint(width), random.randint(height))
        if grid[x, y, :].any():
            continue
        pos = (x, y)
    return pos


def neighbors(pos, width, height, dir=None):
    x, y = pos
    neighbors = []
    if x > 0 and (dir is None or dir == LEFT):
        neighbors.append((x-1, y))
    if y > 0 and (dir is None or dir == DOWN):
        neighbors.append((x, y-1))
    if x < width - 1 and (dir is None or dir == RIGHT):
        neighbors.append((x+1, y))
    if y < height - 1 and (dir is None or dir == UP):
        neighbors.append((x, y+1))
    return neighbors


def dir_to_str(dir):
    """
    UP and DOWN are flipped in visualization
    """
    dir_dict = {0: 'UP', 1: 'DOWN', 2: 'LEFT', 3: 'RIGHT', 4: 'USE'}
    return dir_dict[dir]


class CraftWorld(object):
    """
    Information of the world
    """
    def __init__(self, config):
        self.cookbook = Cookbook(config.recipes)
        self.switch_init_pos = config.world.switch_init_pos
        self.width = config.world.width
        self.height = config.world.height
        self.window_width = config.world.win_width
        self.window_height = config.world.win_height
        self.n_workshops = config.world.n_workshops
        self.num_res = config.world.num_res.__dict__ if 'num_res' in config.world.__dict__.keys() else None

        self.n_features = \
            2 * self.window_width * self.window_height * self.cookbook.n_kinds + \
            self.cookbook.n_kinds + \
            4 + \
            1
        self.n_actions = N_ACTIONS

        self.non_grabbable_indices = self.cookbook.environment
        self.grabbable_indices = [i for i in range(self.cookbook.n_kinds)
                                  if i not in self.non_grabbable_indices]
        self.workshop_indices = [self.cookbook.index["workshop%d" % i]
                                 for i in range(self.n_workshops)]
        self.water_index = self.cookbook.index["water"]
        self.stone_index = self.cookbook.index["stone"]

        self.random = np.random.RandomState(2)

    def sample_scenario_with_goal(self, goal):
        """
        goal: idx (str_goal -> idx: cookbook.index[str_goal])
        """
        assert goal not in self.cookbook.environment
        if goal in self.cookbook.primitives:
            make_island = goal == self.cookbook.index["gold"]
            make_cave = goal == self.cookbook.index["gem"]
            return self.sample_scenario({goal: 1}, make_island=make_island,
                                        make_cave=make_cave)
        elif goal in self.cookbook.recipes:
            ingredients = self.cookbook.primitives_for(goal)
            return self.sample_scenario(ingredients)
        else:
            assert False, "don't know how to build a scenario for %s" % goal

    def sample_scenario(self, ingredients, make_island=False, make_cave=False):
        # generate grid
        grid = np.zeros((self.width, self.height, self.cookbook.n_kinds))
        i_bd = self.cookbook.index["boundary"]
        grid[0, :, i_bd] = 1
        grid[self.width-1:, :, i_bd] = 1
        grid[:, 0, i_bd] = 1
        grid[:, self.height-1:, i_bd] = 1

        # treasure
        if make_island or make_cave:
            (gx, gy) = (1 + np.random.randint(self.width-2), 1)
            treasure_index = \
                self.cookbook.index["gold"] if make_island else self.cookbook.index["gem"]
            wall_index = \
                self.water_index if make_island else self.stone_index
            grid[gx, gy, treasure_index] = 1
            for i in range(-1, 2):
                for j in range(-1, 2):
                    if not grid[gx+i, gy+j, :].any():
                        grid[gx+i, gy+j, wall_index] = 1

        # ingredients
        if self.num_res is None:
            for primitive in self.cookbook.primitives:
                if primitive == self.cookbook.index["gold"] or \
                        primitive == self.cookbook.index["gem"]:
                    continue
                for i in range(4):
                    (x, y) = random_free(grid, self.random, self.width, self.height)
                    grid[x, y, primitive] = 1
        else:  # num of resources defined by config
            for k, v in self.num_res.items():
                primitive = self.cookbook.index[k]
                for i in range(v):
                    (x, y) = random_free(grid, self.random, self.width, self.height)
                    grid[x, y, primitive] = 1

        # generate crafting stations
        for i_ws in range(self.n_workshops):
            ws_x, ws_y = random_free(grid, self.random, self.width, self.height)
            workshop_idx = self.cookbook.index["workshop%d" % i_ws]
            if workshop_idx is None:
                continue
            grid[ws_x, ws_y, workshop_idx] = 1

        # generate init pos
        init_pos = random_free(grid, self.random, self.width, self.height)

        return CraftScenario(grid, init_pos, self)

    def visualize(self, transitions):
        def _visualize(win):
            curses.start_color()
            for i in range(1, 8):
                curses.init_pair(i, i, curses.COLOR_BLACK)
                curses.init_pair(i+10, curses.COLOR_BLACK, i)
            states = [transitions[0].s1] + [t.s2 for t in transitions]
            mstates = [transitions[0].m1] + [t.m2 for t in transitions]
            for state, mstate in zip(states, mstates):
                win.clear()
                for y in range(self.height):
                    for x in range(self.width):
                        if not (state.grid[x, y, :].any() or (x, y) == state.pos):
                            continue
                        thing = state.grid[x, y, :].argmax()
                        if (x, y) == state.pos:
                            if state.dir == LEFT:
                                ch1 = "<"
                                ch2 = "@"
                            elif state.dir == RIGHT:
                                ch1 = "@"
                                ch2 = ">"
                            elif state.dir == UP:
                                ch1 = "^"
                                ch2 = "@"
                            elif state.dir == DOWN:
                                ch1 = "@"
                                ch2 = "v"
                            color = curses.color_pair(mstate.arg or 0)
                        elif thing == self.cookbook.index["boundary"]:
                            ch1 = ch2 = curses.ACS_BOARD
                            color = curses.color_pair(10 + thing)
                        else:
                            name = self.cookbook.index.get(thing)
                            ch1 = name[0]
                            ch2 = name[-1]
                            color = curses.color_pair(10 + thing)

                        win.addch(self.height-y, x*2, ch1, color)
                        win.addch(self.height-y, x*2+1, ch2, color)
                win.refresh()
                time.sleep(1)
        curses.wrapper(_visualize)


class CraftScenario(object):
    """
    Init state of the world
    """
    def __init__(self, grid, init_pos, world):
        self.init_grid = grid
        self.init_pos = init_pos
        self.init_dir = 0
        self.world = world

    def init(self):
        inventory = np.zeros(self.world.cookbook.n_kinds)
        if self.world.switch_init_pos:
            self.switch_init_pos()
        state = CraftState(self, self.init_grid,
                           self.init_pos, self.init_dir, inventory)
        return state
    
    def switch_init_pos(self):
        self.init_pos = random_free(self.init_grid, self.world.random, self.world.width, self.world.height)

    def __str__(self):
        out = np.argmax(self.init_grid, axis=2)
        out[self.init_pos] = -1
        return np.array_str(out.T)


class CraftState(object):
    """
    Environmental state
    """
    def __init__(self, scenario, grid, pos, dir, inventory):
        self.scenario = scenario
        self.world = scenario.world
        self.grid = grid
        self.inventory = inventory
        self.pos = pos
        self.dir = dir
        self._cached_features = None

    def satisfies(self, goal_name, goal_arg):
        return self.inventory[goal_arg] > 0

    def features(self):
        if self._cached_features is None:
            x, y = self.pos
            hw = self.world.window_width // 2
            hh = self.world.window_height // 2
            bhw = (self.world.window_width * self.world.window_width) // 2
            bhh = (self.world.window_height * self.world.window_height) // 2

            grid_feats = array.pad_slice(self.grid, (x-hw, x+hw+1),
                                         (y-hh, y+hh+1))
            grid_feats_big = array.pad_slice(self.grid, (x-bhw, x+bhw+1),
                                             (y-bhh, y+bhh+1))
            grid_feats_big_red = block_reduce(grid_feats_big,
                                              (self.world.window_width, self.world.window_height, 1), func=np.max)

            self.gf = grid_feats.transpose((2, 0, 1))
            self.gfb = grid_feats_big_red.transpose((2, 0, 1))

            pos_feats = np.asarray(self.pos)
            pos_feats[0] /= self.world.width
            pos_feats[1] /= self.world.height

            dir_features = np.zeros(4)
            dir_features[self.dir] = 1  # direction

            features = np.concatenate((grid_feats.ravel(),
                                       grid_feats_big_red.ravel(), self.inventory,
                                       dir_features, [0]))
            assert len(features) == self.world.n_features
            self._cached_features = features

        return self._cached_features

    def step(self, action):
        x, y = self.pos
        n_dir = self.dir
        n_inventory = self.inventory
        n_grid = self.grid

        reward = 0

        # move actions
        if action == DOWN:
            dx, dy = (0, -1)
            n_dir = DOWN
        elif action == UP:
            dx, dy = (0, 1)
            n_dir = UP
        elif action == LEFT:
            dx, dy = (-1, 0)
            n_dir = LEFT
        elif action == RIGHT:
            dx, dy = (1, 0)
            n_dir = RIGHT

        # use actions
        elif action == USE:
            cookbook = self.world.cookbook
            dx, dy = (0, 0)
            success = False
            for nx, ny in neighbors(self.pos, self.world.width, self.world.height, self.dir):
                here = self.grid[nx, ny, :]
                if not self.grid[nx, ny, :].any():  # is empty
                    continue

                if here.sum() > 1:
                    logging.error("impossible world configuration:")
                    logging.error(here.sum())
                    logging.error(self.grid.sum(axis=2))
                    logging.error(self.grid.sum(axis=0).sum(axis=0))
                    logging.error(cookbook.index.contents)
                assert here.sum() == 1
                thing = here.argmax()

                if not(thing in self.world.grabbable_indices or
                        thing in self.world.workshop_indices or
                        thing == self.world.water_index or
                        thing == self.world.stone_index):
                    continue

                n_inventory = self.inventory.copy()
                n_grid = self.grid.copy()

                if thing in self.world.grabbable_indices:
                    n_inventory[thing] += 1
                    n_grid[nx, ny, thing] = 0
                    success = True

                elif thing in self.world.workshop_indices:
                    # TODO not with strings
                    workshop = cookbook.index.get(thing)
                    for output, inputs in cookbook.recipes.items():
                        # make any available tools
                        if inputs["_at"] != workshop:
                            continue
                        yld = inputs["_yield"] if "_yield" in inputs else 1
                        ing = [i for i in inputs if isinstance(i, int)]
                        if any(n_inventory[i] < inputs[i] for i in ing):
                            continue
                        n_inventory[output] += yld
                        for i in ing:
                            n_inventory[i] -= inputs[i]
                        success = True

                elif thing == self.world.water_index:
                    if n_inventory[cookbook.index["bridge"]] > 0:
                        n_grid[nx, ny, self.world.water_index] = 0
                        n_inventory[cookbook.index["bridge"]] -= 1

                elif thing == self.world.stone_index:
                    if n_inventory[cookbook.index["axe"]] > 0:
                        n_grid[nx, ny, self.world.stone_index] = 0

                break

        # other
        else:
            raise Exception("Unexpected action: %s" % action)

        n_x = x + dx
        n_y = y + dy
        if self.grid[n_x, n_y, :].any():
            n_x, n_y = x, y

        new_state = CraftState(self.scenario, n_grid,
                               (n_x, n_y), n_dir, n_inventory)
        return reward, new_state

    def next_to(self, i_kind):
        x, y = self.pos
        return self.grid[x-1:x+2, y-1:y+2, i_kind].any()
    
    def __str__(self):
        map = np.argmax(self.grid, axis=2)
        map[self.pos] = -1
        str = '\nMap:\n' + np.array_str(map.T)
        str += '\nBag:\n' + np.array_str(self.inventory)
        return str
