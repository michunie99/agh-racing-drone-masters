from dataclasses import dataclass
from pathlib import Path
from typing import List, Tuple
import csv

import numpy as np
import pybullet as p

from .utils import calculateRelativeObseration

@dataclass
class Gate:
    position: np.array
    normal: np.array
    quat: np.array
    scale: np.array
    asset: str

class Segment():
    def __init__(
            self,
            start_gate: Gate,
            end_gate: Gate
            ):
        self.start_gate=start_gate
        self.end_gate=end_gate
        diff_vec=end_gate.position-start_gate.position
        self.vec=diff_vec/np.lialg.norm(diff_vec)

    def project_point(
            self,
            point: np.array
            ) -> np.array:
        point=point-self.start_gate.position
        return np.dot(point, self.vec)

    def completed(
            self,
            point: np.array
            ) -> bool:
        diff_vec=point-self.end_gate.position
        proj=np.dot(diff_vec, self.end_gate.normal)
        return proj <= 0

class Track():
    def __init__(
            self,
            track: List[Segment]
            ):
        self.track=track

    def update(
            self,
            point: np.array,
            curr_seg: int
            ) -> Tuple[bool, np.array]:
        segment=self.track(curr_seg)
        projection=segment.project_point(point)
        completed=segment.completed(point)
        return completed, projection


class TrackLoader():
    def __init__(
            self,
            asset_path: Path
            ):
        self.asset_path=asset_path

    def _parserCSV(self):
        with open(self.track_path) as file:
            reader = csv.reader(file)
            track_data = list(reader)
        return track_data

    def _projectNorm(self, quat):
        unit=np.array([-1,0,0])
        norm = p.multiplyTransforms(
                unit,
                ort
        )
        return norm

    def parseTrack(
            self,
            track_path: Path,
            ):
        track_data=self._parserCSV(track_path)
        track=[]
         
        for gate in track_data:
            asset=asset_path / np.array(gate[0])
            pos=np.array(gate[1],gate[2],gate[3])
            quat=np.array(gate[4],gate[5],gate[6],gate[7])
            scale=gate[8]
            
            gate=Gate(
                    pos, 
                    self._projectNorm(quat),
                    quat,
                    scale,
                    asset
            )
            track.append(gate)

        return track

    def loadBullet(
            self,
            track: List[Gate],
            clientID: int
        ):
        g_urdfs = []
        for gate in track:
            g_urdf  = p.loadURDF(   str(gate.asset),
                                    gate.pos, 
                                    gate.ort, 
                                    globalScaling=gate.scale,
                                    useFixedBase=1,
                                    physicsClientId=CLIENT,
                            )
            g_urdfs.append(g_urdf)

        self.g_urdfs=g_urdfs

    def genObs(
            self,
            track: List[Gate]
        ):
        observations = []
        for g1, g2 in zip(track[:-1], track[1:]):
            obj1, obj2 = g1.position, g2.position
            observations.append(calculaterelativeobseration(obj1, obj2))
        
        obj1 = track[-1].position
        obj2 = track[0].position
        observations.append(calculateRelativeObseration(obj1, obj2))

        self.observations=observations

    def createTrack(
            self,
            track: List[Gate]
        ):
        segments = []
        for g1, g2 in zip(track[:-1], track[1:]):
            segments.append(Segment(g1,g2))

        g1 = track[-1]
        g2 = track[0]
        segments.append(Segment(g1,g2))
        
        self.track=Track(segments)
        

if __name__ == '__main__':
    track_path='tracks/cricle_track.csv'
    clientID=p.connect(p.GUI)
    race_track=TrackLoader('assets')
    track=race_track.parseTrack(track_path)
    race_track.genObs(track)
    race_track.loadBullet(track, clientID)
    race_track.createTrack(track)

    input()


