from pathlib import Path
import csv
from typing import Union
from collections import deque
import time

import pybullet as p
import numpy as np

from src.utils import calculateRelativeObseration

class TrackLoader():
    def __init__(self,
                 physicsClientId:int,
                 track_path: Union[str, Path] = "tracks/sample_track.csv",
                 assets_path: Union[str, Path] = "assets/",
                 randomise: bool = False,
                 seed: int = 1,
                 ):
        np.random.seed(seed)

        self.track_path = Path(track_path)
        self.assets_path = Path(assets_path)
        self.randomise = randomise
        self.CLIENT = physicsClientId

    def _readCSV(self):
        with open(self.track_path) as file:
            reader = csv.reader(file)
            track_data = list(reader)
        return track_data
    
    def _parseTrack(self):
        track_data = self._readCSV()
        
        track = []
        for gate in track_data:
            #TODO - add randomization
            asset_path = self.assets_path / gate[0]
            pos = (float(gate[1]), float(gate[2]), float(gate[3]))
            ort = ( float(gate[4]), float(gate[5]),
                    float(gate[6]), float(gate[7]))
            scale = float(gate[8])
            
            track.append((asset_path, np.array(pos), 
                          np.array(ort), scale))

        return track
            
    def _generateObservations(self, track):
        observations = deque(maxlen = len(track) + 1)
        observations.append(([0, 0, 0], 0))
        for g1, g2 in zip(track[:-1], track[1:]):
            obj1, obj2 = g1[1:3], g2[1:3]
            observations.append(calculateRelativeObseration(obj1, obj2))

        return observations

    def _addTrackBullet(self, track):
        g_urdfs = []
        for g in track:
            asset_path, pos, ort, scale = g
            g_urdf  = p.loadURDF(   str(asset_path),
                                    pos, 
                                    ort, 
                                    globalScaling=scale,
                                    useFixedBase=0,
                                    physicsClientId=self.CLIENT,
                                    flags=p.URDF_MERGE_FIXED_LINKS

                            )
            g_urdfs.append(g_urdf)
            # print(g_urdf)

        return g_urdfs

    def loadTrack(self):
        track = self._parseTrack()
        obs = self._generateObservations(track)
        g_urdfs = self._addTrackBullet(track)

        return obs, g_urdfs
    
if __name__ == "__main__":
    import pybullet_data

    physicClient = p.connect(p.GUI)
    p.setAdditionalSearchPath(pybullet_data.getDataPath())
    p.setAdditionalSearchPath(pybullet_data.getDataPath())
    p.setGravity(0,0,0)
    planeId = p.loadURDF("plane.urdf")

    loader = TrackLoader(physicClient, "tracks/single_gate.csv")
    obs, g_urdfs = loader.loadTrack()

    startPos = [0, 0, 1]
    startOrientation = p.getQuaternionFromEuler([0,0,0])
    racer = p.loadURDF("assets/racer.urdf",startPos, startOrientation, globalScaling=1)
    try:
        while True:
            p.stepSimulation()
            time.sleep(1./240.)

    except KeyboardInterrupt:
        p.disconnect()
        print(obs)