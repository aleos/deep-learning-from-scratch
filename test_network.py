import numpy as np
import pytest

from network import Network, sigmoid


class TestSigmoid:
    def test_zero_returns_half(self) -> None:
        assert sigmoid(np.array([0.0])) == 0.5

    def test_large_positive_approaches_one(self) -> None:
        assert sigmoid(np.array([100.0])) > 0.99

    def test_large_negative_approaches_zero(self) -> None:
        assert sigmoid(np.array([-100.0])) < 0.01


class TestNetwork:
    def test_output_shape(self) -> None:
        net = Network([784, 16, 16, 10])
        output = net.feedforward(np.random.randn(784, 1))
        assert output.shape == (10, 1)

    def test_output_between_zero_and_one(self) -> None:
        net = Network([784, 16, 16, 10])
        output = net.feedforward(np.random.randn(784, 1))
        assert np.all(output >= 0.0)
        assert np.all(output <= 1.0)

    @pytest.mark.parametrize(
        "sizes",
        [
            [784, 16, 16, 10],
            [4, 3, 2],
            [10, 5, 5, 5, 3],
        ],
    )
    def test_output_matches_last_layer(self, sizes: list[int]) -> None:
        net = Network(sizes)
        output = net.feedforward(np.random.randn(sizes[0], 1))
        assert output.shape == (sizes[-1], 1)
