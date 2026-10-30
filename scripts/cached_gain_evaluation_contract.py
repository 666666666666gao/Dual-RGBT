"""Split admission and fixed LasHeR sequence/frame totals."""


def scope(split, smoke_frames, exploratory_test, full_evaluation_allowed):
    if split == 'test':
        assert smoke_frames == 0
        assert exploratory_test is True
        assert full_evaluation_allowed is True
        return 245, 220703, True
    assert split == 'dev'
    return 97, 54646, False
