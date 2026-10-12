from regression import Regressor


class TestRegression:

    def test_gradient_descent(self):
        c = Regressor()
        c.create_dataset()
        before, _ = c.calc_loss()
        c.gradient_descent(iterations=20)
        after, _ = c.calc_loss()
        assert after.data < before.data
