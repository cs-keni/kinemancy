from src.constants import GestureLabel, STATIC_GESTURES, DYNAMIC_GESTURES, GestureEvent


def test_gesture_label_static_range():
    assert GestureLabel.OPEN_PALM == 0
    assert GestureLabel.NONE == 8


def test_gesture_label_dynamic_range():
    assert GestureLabel.SNAP == 9
    assert GestureLabel.CLAP == 15


def test_all_values_distinct():
    values = [int(g) for g in GestureLabel]
    assert len(values) == len(set(values))


def test_int_comparison():
    assert int(GestureLabel.FIST) == 1
    assert GestureLabel.NONE > GestureLabel.OPEN_PALM


def test_static_dynamic_disjoint():
    assert STATIC_GESTURES.isdisjoint(DYNAMIC_GESTURES)


def test_static_dynamic_cover_all():
    all_labels = set(GestureLabel)
    assert all_labels == STATIC_GESTURES | DYNAMIC_GESTURES


def test_gesture_event_fields():
    import time
    evt = GestureEvent(
        label=GestureLabel.SNAP,
        confidence=0.95,
        timestamp=time.time(),
        hand_x=0.5,
        hand_y=0.4,
    )
    assert evt.label == GestureLabel.SNAP
    assert 0.0 <= evt.confidence <= 1.0
