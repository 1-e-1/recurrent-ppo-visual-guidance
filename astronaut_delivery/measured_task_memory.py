"""Measured task queue and conservative map-history replay; no scene oracle."""
import numpy as np


def replay_changes(mapper,events):
    """An initial scan MUST lose all superseded evidence before map reuse.

    Deliberately does not recover intervening free measurements. Replaying just
    these invalidations is conservative and cannot resurrect old free cells.
    """
    count=0
    for event in events:
        if event.get('event')!='local_evidence_retired_to_UNKNOWN':continue
        mapper.invalidate_region(event['lower'],event['upper'],**event.get('physical_evidence_bounds',{}))
        count+=1
    return count


class MeasuredTaskMemory:
    def __init__(self):
        self.seen={};self.completed=[];self.used_places=[]

    def observe(self,estimates):
        for estimate in estimates.values():
            if estimate.kind=='magnetic_interface' and estimate.name not in self.completed:
                previous=self.seen.get(estimate.name)
                if previous is None or estimate.stamp>previous.stamp:self.seen[estimate.name]=estimate

    def pending(self):return [e for name,e in self.seen.items() if name not in self.completed]

    def place_available(self,position):
        return all(np.linalg.norm(np.asarray(position)[:2]-p[:2])>.070 for p in self.used_places)

    def complete(self,name,place):
        if name not in self.seen or name in self.completed:raise ValueError('unobserved or duplicate task completion')
        self.completed.append(name);self.used_places.append(np.asarray(place,float).copy())
