from env import *

AGENT_VIEW = 5

def make_config():
  # specify the item types
    items = []
    items.append(Item("JellyBean",     [0, 0, 1.0], [0, 0, 1.0], [0, 0, 0], [0, 0, 0], False, 0.0,
            intensity_fn=IntensityFunction.CONSTANT, intensity_fn_args=[-3.5],
            interaction_fns=[
              [InteractionFunction.PIECEWISE_BOX, 3, 10, 1, -2],
              [InteractionFunction.ZERO],
              [InteractionFunction.PIECEWISE_BOX, 25,50,-50,-10]
            ]))
    items.append(Item("Banana",    [0.0, 1.0, 0.0], [0.0, 1.0, 0.0], [0, 0, 0], [0, 0, 0], False, 0.0,
            intensity_fn=IntensityFunction.CONSTANT, intensity_fn_args=[-6.0],
            interaction_fns=[
              [InteractionFunction.ZERO],
              [InteractionFunction.ZERO],
              [InteractionFunction.ZERO]
            ]))
    items.append(Item("Onion", [1.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0, 0, 0], [0, 0, 0], False, 0.0,
            intensity_fn=IntensityFunction.CONSTANT, intensity_fn_args=[-3.5],
            interaction_fns=[
              [InteractionFunction.PIECEWISE_BOX, 25,50,-50,-10],
              [InteractionFunction.ZERO],
              [InteractionFunction.PIECEWISE_BOX, 3, 10, 1, -2]
            ]))
  # construct the simulator configuration
    return SimulatorConfig(max_steps_per_movement=1, vision_range=AGENT_VIEW,
        allowed_movement_directions=[ActionPolicy.ALLOWED, ActionPolicy.ALLOWED, ActionPolicy.ALLOWED,
                                     ActionPolicy.ALLOWED],
        allowed_turn_directions=[ActionPolicy.DISALLOWED, ActionPolicy.DISALLOWED, ActionPolicy.DISALLOWED,
                                 ActionPolicy.DISALLOWED],
        no_op_allowed=False, patch_size=32, mcmc_num_iter=4000, items=items,
        agent_color=[0.0, 0.0, 0.0], agent_field_of_view=2*pi,
        collision_policy=MovementConflictPolicy.FIRST_COME_FIRST_SERVED,
        decay_param=0.0, diffusion_param=0.14, deleted_item_lifetime=500)

def make_reward():
    def get_reward(prev_item, item, T):
        if (T//150000)%2 == 0:
            if item[0] - prev_item[0] == 1:
                return 2
            elif item[2] - prev_item[2] == 1:
                return -1
            elif item[1] - prev_item[1] == 1:
                return 0.1
            else:
                return 0
        else:
            if item[0] - prev_item[0] == 1:
                return -1
            elif item[2] - prev_item[2] == 1:
                return 2
            elif item[1] - prev_item[1] == 1:
                return 0.1
            else:
                return 0
    return get_reward