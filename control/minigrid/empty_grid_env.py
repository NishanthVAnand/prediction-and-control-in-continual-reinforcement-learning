from gym_minigrid.minigrid import *
from gym.spaces import Discrete

class LGrid(MiniGridEnv):
	
	def __init__(self, goal_pos_rew={(11, 1):+5, (13, 3):0}, seed=None, agent_start_pos=None):
		self.goal_pos_rew = goal_pos_rew
		self.agent_start_pos = agent_start_pos
		self.wall_positions = [(3, 4), (4, 4)]

		super().__init__(
			width=15,
			height=11,
			max_steps=100,
			see_through_walls=False,
			agent_view_size=5,
			seed=seed,
			mission_space=MissionSpace(mission_func=lambda: "get to the correct green goal square")
			)

		# Allow only 3 actions permitted: left, right, forward
		self.action_space = Discrete(self.Actions.forward + 1)

	def _gen_grid(self, width, height):

		# Create an empty grid
		self.grid = Grid(width, height)

		# Generate the surrounding walls
		self.grid.wall_rect(0, 0, width, height)

		# Add goals
		for goal_pos, _ in self.goal_pos_rew.items():
			self.put_obj(Goal(), goal_pos[0], goal_pos[1])

		# Add walls
		for wall_pos in self.wall_positions:
			self.put_obj(Wall(), wall_pos[0], wall_pos[1])

		# Generate walls for L-Grid
		for i in list(range(5, 15)):
			self.grid.vert_wall(i, 4, 7)

		# Place the agent
		if self.agent_start_pos is not None:
			self.agent_pos = self.agent_start_pos
			self.agent_dir = self.agent_start_dir
		else:
			self.agent_pos = self.place_agent(top=(1,9), size=(4,2))

	def step(self, action):
		self.step_count += 1

		reward = 0.0
		terminated = False
		truncated = False

		# Get the position in front of the agent
		fwd_pos = self.front_pos

		# Get the contents of the cell in front of the agent
		fwd_cell = self.grid.get(*fwd_pos)

		# Rotate left
		if action == self.actions.left:
			self.agent_dir -= 1
			if self.agent_dir < 0:
				self.agent_dir += 4

		# Rotate right
		elif action == self.actions.right:
			self.agent_dir = (self.agent_dir + 1) % 4

		# Move forward
		elif action == self.actions.forward:
			if fwd_cell == None or fwd_cell.can_overlap():
				self.agent_pos = fwd_pos
			if fwd_cell != None and fwd_cell.type == 'goal':
				terminated = True
				reward = self.goal_pos_rew[tuple(fwd_pos)]
				
		else:
			assert False, "unknown action"

		if self.step_count >= self.max_steps:
			truncated = True

		obs = self.gen_obs()

		return obs, reward, terminated, truncated