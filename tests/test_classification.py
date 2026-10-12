from classification import Classifier

class TestClassification:
    def test_create_dataset(self):
        c = Classifier()
        c.create_dataset()
        assert set(c.y) == {-1, 1}

    def test_gradient_descent(self):
        c = Classifier()
        c.create_dataset()
        before, _ = c.calc_loss()
        c.gradient_descent(iterations=20)
        after, _ = c.calc_loss()
        assert after.data < before.data