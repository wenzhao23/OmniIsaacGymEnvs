from omni.isaac.kit import SimulationApp

simulation_app = SimulationApp({"headless": False})

from omni.isaac.core import World

my_world = World(stage_units_in_meters=1.0)
