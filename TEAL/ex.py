import torch
from torch import nn
def input_prune(layer, x):
    """
    - batch 내 각 sample마다 서로 다른 pruned weight을 적용해야 함.
    - batch_size 개수만큼 pruned된 nn.Linear 레이어를 생성하여 리스트로 저장.

    Args:
        layer (nn.Linear): 원본 Linear Layer (W: [out_features, in_features])
        x (torch.Tensor): 입력 데이터 (batch_size, sequence_length, hidden_dim)

    Returns:
        pruned_layers (list): batch_size 개수만큼의 pruned nn.Linear 레이어 리스트
    """

    batch_size, sequence_length, hidden_dim = x.shape
    out_features = layer.weight.shape[0]

    # Mask 생성 (sequence 전체에서 한 번이라도 0이면 해당 hidden_dim column을 pruning)
    col_masks = (x == 0).any(dim=1)

    # 각 sample 별 pruned nn.Linear 저장할 리스트 생성
    pruned_layers = []

    # Batch 내 각 sample마다 개별 weight pruning 수행
    with torch.no_grad():
        for batch_idx in range(batch_size):
            # 원본 weight과 bias를 복사하여 새로운 nn.Linear 레이어 생성
            pruned_layer = nn.Linear(hidden_dim, out_features, bias=layer.bias is not None)

            # 원본 weight 복사 (각 sample마다 독립적으로 weight을 가지게 함)
            pruned_layer.weight.data = layer.weight.clone()

            # 현재 sample의 pruning mask 가져오기
            mask = col_masks[batch_idx]

            # mask 적용하여 weight에서 해당 column을 0으로 설정
            pruned_layer.weight.data[:, mask] = 0

            # Bias도 원본에서 복사
            if layer.bias is not None:
                pruned_layer.bias.data = layer.bias.clone()

            # pruned nn.Linear 레이어를 리스트에 추가
            pruned_layers.append(pruned_layer)

    return pruned_layers

torch.manual_seed(42)

# Custom Linear Layer
linear = nn.Linear(in_features=8, out_features=4, bias=True)
linear.weight.data = torch.randn(8, 4)
linear.bias.data = torch.randn(4)

# 입력 데이터 (batch_size=2, sequence_length=2560, hidden_dim=4096)
x = torch.randn(2, 2560, 4096)

# 일부 hidden_dim 위치를 0으로 설정
x[:, :, [100, 500, 1000]] = 0

print("Original Linear Layer:\n", linear)

# Forward 연산 전에 pruning 적용
pruned_layers = input_prune(linear, x)

# 출력 결과 확인
for i, pruned_layer in enumerate(pruned_layers):
    print(f"\nPruned nn.Linear Layer for Sample {i+1}:")
    print(pruned_layer)
    print("Pruned Weight:\n", pruned_layer.weight)