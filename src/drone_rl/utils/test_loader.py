import pybullet as p
import time
import pybullet_data
import csv
import math
import numpy as np

physicClient = p.connect(p.GUI)
p.setAdditionalSearchPath(pybullet_data.getDataPath())

print(pybullet_data.getDataPath())
# /home/michunie/.pyenv/versions/3.10.3/envs/drones/lib/python3.10/site-packages/pybullet_data

p.setGravity(0,0,0)
planeId = p.loadURDF("plane.urdf")

track  = []
with open('tracks/sample_track.csv') as file:
    track_data = csv.reader(file)
    for row in track_data:
        startPos = (float(row[1]),
                    float(row[2]),
                    float(row[3]))
        
        # startOrientation = (float(row[4]),
        #                     float(row[5]),
        #                     float(row[6]),
        #                     float(row[7]))
        startOrientation = p.getQuaternionFromEuler([math.pi/2,0,0])
        glabalSacling = float(row[8])

        gate  = p.loadURDF(         'assets/' +row[0],
                                    startPos, 
                                    startOrientation, 
                                    globalScaling=glabalSacling,
                                    useFixedBase=1,
                                    flags=p.URDF_MERGE_FIXED_LINKS
                                    )
        track.append(gate)
# startPos = [0, 0, 0]

# startOrientation = p.getQuaternionFromEuler([0,0,0])

# gate  = p.loadURDF(  "assets/gate.urdf",
#                             startPos, 
#                             startOrientation, 
#                             globalScaling=1,
#                             useFixedBase=1
#                             )
startPos = [1, 1, 1]
startOrientation = p.getQuaternionFromEuler([0,0,0])
racer = p.loadURDF("assets/racer.urdf",startPos, startOrientation, globalScaling=1)

# print( n := p.getNumJoints(boxId))

# for i in range(n):
#     print(p.getNumJoints(boxId, i))

try:
    while True:
        p.stepSimulation()
        # print(p.getContactPoints())
        gate_cord = []
        for gate in track:
            pos, ort = p.getBasePositionAndOrientation(gate)
            x, y, z = pos
            roll, pitch, yaw = p.getEulerFromQuaternion(ort)
            gate_cord.append((pos, ort))
            # print(f"x: {x}, y: {y}, z: {z}, r: {roll}, p: {pitch}, y: {yaw}")
     
        time.sleep(1./240.)

except KeyboardInterrupt:
    cubePos, cubeOrn = p.getBasePositionAndOrientation(boxId)
    print(cubePos, cubeOrn)
    p.disconnect()

