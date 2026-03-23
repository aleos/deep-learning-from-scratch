import numpy as np

from network import Network

net = Network([784, 16, 16, 10])

# Create a fake "image" - 784 random pixel values
fake_image = np.random.randn(784, 1)

# Feed the fake image through the network
output = net.feedforward(fake_image)

print("Output shape:", output.shape)
print("Output values:", output)
print("Predicted digit:", np.argmax(output))
