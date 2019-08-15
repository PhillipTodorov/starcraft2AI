import sc2
from examples.protoss.cannon_rush import CannonRushBot
from sc2 import run_game, maps, Race, Difficulty
from sc2.player import Bot, Computer
from sc2.constants import NEXUS, PROBE, PYLON, ASSIMILATOR, GATEWAY, CYBERNETICSCORE, STALKER, \
STARGATE, VOIDRAY
import random
import cv2
import numpy as np


class SentdeBot(sc2.BotAI):
	def __init__(self):
		self.ITERATIONS_PER_MINUTE = 165
		self.MAX_WORKERS = 65


	async def on_step(self, iteration):
		self.iteration = iteration 
		await self.distribute_workers()
		await self.build_workers()
		await self.build_pylons()
		await self.build_assimilators()
		await self.expand()
		await self.offensive_force_buildings()
		await self.build_offensive_forces()
		await self.attack()
		await self.intel()

	async def intel(self):
		"""creates a array of zeros of map_size dimentions, and of data type uint8"""
		game_data = np.zeros((self.game_info.map_size[1], self.game_info.map_size[0], 3), np.uint8)
		"""for each nexus, assign onto game_data a circle at coordinates nex_pos[0]"""
		for nexus in self.units(NEXUS):
			nex_pos = nexus.position
			cv2.circle(game_data, (int(nex_pos[0]), int(nex_pos[1])), 10, (0, 255, 0), -1)

		flipped = cv2.flip(game_data, 0)
		resized = cv2.resize(flipped, dsize=None, fx=2, fy=2)
		cv2.imshow('intel',resized)
		cv2.waitKey(1)

	"""builds workers when resources are available"""
	async def build_workers(self):
		if len(self.units(NEXUS))*16 > len(self.units(PROBE)):
			if len(self.units(PROBE)) < self.MAX_WORKERS:
				for nexus in self.units(NEXUS).ready.noqueue:
					if self.can_afford(PROBE):
						await self.do(nexus.train(PROBE))

	"""builds pylons given that there are less than X already built and 1 isnt in the process of being built"""
	async def build_pylons(self):
		if self.supply_left < 5 and not self.already_pending(PYLON):
			nexuses = self.units(NEXUS).ready
			if nexuses.exists:
				if self.can_afford(PYLON):
					await self.build(PYLON, near=nexuses.first)

	"""builds assimilators if vespene geysers are closer than X to the nexus and there is a worker within Y distance"""
	async def build_assimilators(self):
		for nexus in self.units(NEXUS).ready:
			vespenes = self.state.vespene_geyser.closer_than(15.0, nexus)
			for vespene in vespenes:
				if not self.can_afford(ASSIMILATOR):
					break
				worker = self.select_build_worker(vespene.position)
				if worker is None:
					break
				if not self.units(ASSIMILATOR).closer_than(1.0, vespene).exists:
					await self.do(worker.build(ASSIMILATOR, vespene))

	"""build a nexus if there are less than X amount"""
	async def expand(self):
		if self.units(NEXUS).amount < (self.iteration / self.ITERATIONS_PER_MINUTE) and self.can_afford(NEXUS):
			await self.expand_now()

	"""builds cybernetics core given that a pylon and gateway exist and one isnt already pending"""
	async def offensive_force_buildings(self):
		if self.units(PYLON).ready.exists:
			pylon = self.units(PYLON).ready.random

			"""if gateway exists and no cybernetics core and we can afford a cybernetics core and one isnt pending, build cybernetics core. 
			else if the amount of gateways is less than half the amount of minutes elapsed, build gateway"""
			if self.units(GATEWAY).ready.exists and not self.units(CYBERNETICSCORE):
				if self.can_afford(CYBERNETICSCORE) and not self.already_pending(CYBERNETICSCORE):
					await self.build(CYBERNETICSCORE, near=pylon)
			elif len(self.units(GATEWAY)) < 1:
				if self.can_afford(GATEWAY) and not self.already_pending(GATEWAY):
					await self.build(GATEWAY, near=pylon)

			"""if cybernetics core exists and the amount of stargates is less than half the amount of minutes elapsed, build stargate"""		
			if self.units(CYBERNETICSCORE).ready.exists:
				if len(self.units(STARGATE)) < (self.iteration / self.ITERATIONS_PER_MINUTE):
					if self.can_afford(STARGATE) and not self.already_pending(STARGATE):
						await self.build(STARGATE, near=pylon)

	"""builds stalker units"""
	async def build_offensive_forces(self):

		"""if there are no stargates in queue and you can afford to, build voidray"""
		for sg in self.units(STARGATE).ready.noqueue:
			if self.can_afford(VOIDRAY) and self.supply_left > 0:
				await self.do(sg.train(VOIDRAY))

	"""looks for visible enemy units. if none found, head towards known base"""
	def find_target(self,state):
		if len(self.known_enemy_units) > 0:
			return random.choice(self.known_enemy_units)
		elif len(self.known_enemy_structures) > 0:
			return random.choice(self.known_enemy_structures)
		else:
			return self.enemy_start_locations[0]

	"""if you have X amount of stalkers, respond to any visible enemy units, if you have more than Y stalkers, 
	find target and attack"""
	async def attack(self):
		# {UNIT: [n to fight, n to defend]}
		aggressive_units = {VOIDRAY: [8, 3]}

		for UNIT in aggressive_units:
			"""if the number of UNITS are bigger than the amount assigned for both defence and attack, then attack.else, if the number of UNITS is bigger than the number assigned for defence, then defend"""
			if ((self.units(UNIT).amount > aggressive_units[UNIT][0]) and (self.units(UNIT).amount > aggressive_units[UNIT][1])):
				for s in self.units(UNIT).idle:
					await self.do(s.attack(self.find_target(self.state)))
			else:
				if self.units(UNIT).amount > aggressive_units[UNIT][1]:
					if len(self.known_enemy_units) > 0:
						for s in self.units(UNIT).idle:
							await self.do(s.attack(random.choice(self.known_enemy_units)))	



run_game(maps.get("AbyssalReefLE"), [
	Bot(Race.Protoss, SentdeBot()),
	Bot(Race.Protoss, CannonRushBot()),
	], realtime=False)