import time

from poppy_humanoid import PoppyHumanoid
from pypot.vrep.io import VrepIO
from pypot.vrep.remoteApiBindings.vrep import simxSynchronous, simxSynchronousTrigger

vrep = VrepIO("127.0.0.1", 19997)

poppy = PoppyHumanoid(simulator="vrep", shared_vrep_io=vrep)
simxSynchronous(vrep.client_id, True)

while True:
    time.sleep(1)
    simxSynchronousTrigger(vrep.client_id)
