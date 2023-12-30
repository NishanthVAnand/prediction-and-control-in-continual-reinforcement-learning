from gym_minigrid.minigrid import *
from gym.spaces import Discrete

class ItemEnv(MiniGridEnv):
	"""
	Environment with items and a goal.
	"""

	def __init__(self, size=11, item_pos_rew_dict={(2, 3):("red", -2),\
												  (5, 3):("blue", 1),\
												  (3, 5):("yellow", -0.5)},\
										goal_pos_rew_dict={(9, 9):0,\
														   (1, 9):0,\
														   (9, 1):0,\
														   (1, 1):0},\
										seed=None,\
										rew_factor=2,\
										agent_start_pos=None, agent_start_dir=None, vision=5):
		self.item_dict = item_pos_rew_dict
		self.goal_dict = goal_pos_rew_dict
		self.agent_start_pos = agent_start_pos
		self.agent_start_dir = agent_start_dir
		self.rew_factor = rew_factor
		self.rew_dict = {}
		self.walls = [(5,1), (5,2), (5,3), (5,7), (5,8), (5,9),\
						(1,5), (2,5), (3,5), (7,5), (8,5), (9,5)]
		
		super().__init__(
			grid_size=size,
			max_steps=1000,
			# Set this to True for maximum speed
			see_through_walls=False,
			seed=seed,
			agent_view_size=vision,
			mission_space=MissionSpace(mission_func=lambda: "get to the green goal square")
		)

		# Allow only 3 actions permitted: left, right, forward
		self.action_space = Discrete(self.Actions.forward + 1)

	def _gen_grid(self, width, height):
		# Create an empty grid
		self.grid = Grid(width, height)

		# Generate the surrounding walls
		self.grid.wall_rect(0, 0, width, height)
		
		# Place the items
		for pos, (item_col, rew) in self.item_dict.items():
			self.put_obj(Ball(item_col), pos[0], pos[1])
			self.rew_dict[pos] = rew

		# Add walls
		for i, j in self.walls:
			self.put_obj(Wall(), i, j)
			
		# Place the agent
		if self.agent_start_pos is not None:
			self.agent_pos = self.agent_start_pos
			self.agent_dir = self.agent_start_dir
		else:
			self.agent_pos = self.place_agent(top=(3,3), size=(6,6))

		# Place a goal square in the bottom-right corner
		for goal_pos, rew in self.goal_dict.items():
			self.put_obj(Goal(), goal_pos[0], goal_pos[1])

		self.mission = "get to the green goal square"
		
	def step(self, action):
		# Invalid action
		if action >= self.action_space.n:
			raise ValueError(f"Unknown action: {action}")
			
		self.step_count += 1

		reward = 0
		terminated = False
		truncated = False

		# Rotate left
		if action == self.Actions.left:
			self.agent_dir -= 1
			if self.agent_dir < 0:
				self.agent_dir += 4

		# Rotate right
		elif action == self.Actions.right:
			self.agent_dir = (self.agent_dir + 1) % 4

		# Get the position in front of the agent
		fwd_pos = self.front_pos

		# Get the contents of the cell in front of the agent
		fwd_cell = self.grid.get(*fwd_pos)

		# Move forward
		if fwd_cell is None:
			self.agent_pos = tuple(fwd_pos)
		if fwd_cell is not None and fwd_cell.can_pickup():
			self.agent_pos = tuple(fwd_pos)
			self.grid.set(fwd_pos[0], fwd_pos[1], None)
			reward = self.rew_dict[self.agent_pos]
		if fwd_cell is not None and fwd_cell.type == "goal":
			terminated = True
			reward = self.goal_dict[tuple(fwd_pos)]

		if self.step_count >= self.max_steps:
			truncated = True

		obs = self.gen_obs()
		
		return obs, self.rew_factor*reward, terminated, truncated