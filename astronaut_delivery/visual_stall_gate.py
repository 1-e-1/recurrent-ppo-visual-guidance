"""Visual-stall transition gate; no motion or permission authority."""
RECOVERABLE=frozenset(('visual_loss_smooth_brake','visual_quality_smooth_brake',
                       'pre_execution_measurement_rejected'))


def rank_views(left,views):
    """Historical planar-support proxy orders proposals, never grants visibility."""
    from astronaut_delivery.visual_sufficiency import projected_support_proxy
    scored=[(projected_support_proxy(left.camera(view['q'])[:3,3],view['position']),index,view)
            for index,view in enumerate(views)]
    return [view for score,index,view in sorted(scored,key=lambda item:(-item[0],item[1]))]


class VisualStallGate:
    def __init__(self):
        self.owner=None;self.consecutive=0

    def context(self,owner):
        if owner!=self.owner:self.owner=owner;self.consecutive=0

    def stopped(self,owner,reason):
        self.context(owner)
        if reason not in RECOVERABLE:raise ValueError('not a recoverable visual stop')
        self.consecutive+=1

    def completed(self,owner):
        self.context(owner);self.consecutive=0

    def sensing_due(self,owner,*,enabled,at_rest,used_attempts):
        self.context(owner)
        return bool(enabled and at_rest and self.consecutive>=2 and 0<=used_attempts<6)

    def sensing_attempted(self,owner):
        # An unavailable pose cannot cause an unbounded camera retry loop.
        # This counter reset never resets the controller's six-attempt budget.
        self.context(owner);self.consecutive=0


def attempt_recovery(controller,gate,owner):
    """Delegate only to existing map-certified history sensing, right at rest."""
    import numpy as np
    identity=getattr(getattr(controller,'held',None),'name',None) or controller.active
    used=getattr(controller,'history_reacquisition_counts',{}).get(identity,0)
    if not gate.sensing_due(owner,enabled=getattr(controller,'history_reacquisition',False),
                           at_rest=controller.at_known_rest(),used_attempts=used):return False
    before=np.asarray(controller.q[1],float).copy()
    gate.sensing_attempted(owner)
    previous=getattr(controller,'_history_stall_ranking',False)
    controller._history_stall_ranking=True
    try:result=controller.reacquire_from_measured_history()
    finally:controller._history_stall_ranking=previous
    if not np.array_equal(before,controller.q[1]) or not controller.at_known_rest():
        raise RuntimeError('stall_sensing_requires_right_rest')
    controller.decisions.append(dict(status='VISUAL_STALL_HISTORY_SENSING',time_s=controller.time,
        owner=identity,phase=controller.transfer_phase,attempted=bool(result),
        history_attempts_used=getattr(controller,'history_reacquisition_counts',{}).get(identity,0),
        scope='Existing bounded camera-only recovery; fresh feedback still required before right motion'))
    return bool(result)

