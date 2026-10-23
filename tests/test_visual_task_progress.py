from visual_guidance_lab.task_progress import retains_progress,episode_failed


def test_noop_and_backwards_cannot_displace_advancing_baseline():
    assert not retains_progress([0,0,0],[.02,0,0],[0,0,0],[1,0,0])
    assert not retains_progress([0,0,0],[.02,0,0],[-.01,0,0],[1,0,0])
    assert retains_progress([0,0,0],[.02,0,0],[.015,0,0],[1,0,0])


def test_timeout_without_error_is_still_failure():
    assert episode_failed(dict(status='TIMEOUT',error=None))
    assert episode_failed(dict(status='STOPPED',error=None,observed_queue_exhausted=True))
    assert not episode_failed(dict(status='OBSERVED_RGBD_TASK_QUEUE_COMPLETED_PENDING_SCENE_AUDIT',
                                    error=None,observed_queue_exhausted=True))
