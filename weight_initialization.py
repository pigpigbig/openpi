import torch

def initialize_weights(original_weights, geometry):
    original_in = original_weights.shape[0]
    original_out = original_weights.shape[1]
    assert geometry[0] == original_in, "Input dimension mismatch"
    assert geometry[-1] == original_out, "Output dimension mismatch"
    for i in range(1, len(geometry) - 1):
        assert geometry[i] >= 2 * original_in, "Hidden layer size must be at least two times the input dimension"
    output_weights = []
    weight = torch.cat((torch.eye(original_in-1), torch.zeros(original_in-1, geometry[1] - 2*(original_in-1)), -torch.eye(original_in-1)), dim=1)
    weight = torch.cat((weight, torch.zeros(1, geometry[1])), dim=0)
    output_weights.append(weight.T)
    for i in range(1, len(geometry) - 2):
        w1 = torch.cat((torch.eye(original_in-1), torch.zeros(geometry[i] - 2*(original_in-1), original_in-1), -torch.eye(original_in-1), torch.zeros(1, original_in-1)), dim=0)
        w2 = torch.cat((torch.eye(original_in-1), torch.zeros(original_in-1, geometry[i+1] - 2*(original_in-1)), -torch.eye(original_in-1)), dim=1)
        weight = w1 @ w2
        output_weights.append(weight.T)
    w1 = torch.cat((torch.eye(original_in-1), torch.zeros(geometry[-2] - 2*(original_in-1), original_in-1), -torch.eye(original_in-1), torch.zeros(1, original_in-1)), dim=0)
    w1 = torch.cat((w1, torch.zeros(geometry[-2]+1, 1)), dim=1)
    w1[-1, -1] = 1.0
    weight = w1 @ original_weights
    output_weights.append(weight.T)
    return output_weights

# testing
if __name__ == "__main__":
    original_weights = torch.randn(21, 5)
    geometry = [21, 50, 50, 5]
    initialized_weights = initialize_weights(original_weights, geometry)
    # get a test input
    test_input = torch.randn(1, 20)
    # attach bias term to test input
    test_input = torch.cat((test_input, torch.ones(1, 1)), dim=1)
    # forward pass through original weights
    original_output = test_input @ original_weights
    # forward pass through initialized weights
    x = test_input
    for weight in initialized_weights:
        x = weight @ x.T
        # attach bias term if not last layer
        if weight is not initialized_weights[-1]:
            x = torch.cat((x, torch.ones(1, 1)), dim=0)
        x = x.T
        # pass through ReLU if not last layer
        if weight is not initialized_weights[-1]:
            x = torch.relu(x)
    initialized_output = x
    # compare outputs
    print("Original Output:", original_output)
    print("Initialized Output:", initialized_output)
    