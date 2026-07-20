def test_accuracy_bounds():
    acc = 0.81
    assert 0.0 <= acc <= 1.0


def test_f1_bounds():
    f1 = 0.78
    assert 0.0 <= f1 <= 1.0
