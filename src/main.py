import argparse
import torch
from MaskLG_vit import VisionTransformer

def get_masked_parameters(model, num_blocks, epochs, lr, temperature, threshold, device, lambda_sparse=1e-3):
    masked_model = VisionTransformer(
        depth=num_blocks, mask_training=True, temperature=temperature, threshold=threshold
    ).to(device)
    masked_model.load_state_dict(model.state_dict(), strict=False)

    mask_params = [p for n, p in masked_model.named_parameters() if 'mask_logits' in n]
    optimizer = torch.optim.Adam(mask_params, lr=lr)
    dummy_input = torch.randn(2, 197, masked_model.embed_dim).to(device)
    # 真实任务损失可替换为你的数据和标签
    criterion = torch.nn.MSELoss()

    masked_model.train()
    for epoch in range(epochs):
        optimizer.zero_grad()
        out = masked_model.forward_features(dummy_input)
        # 这里用MSELoss做示例，实际应用真实任务损失
        target = torch.zeros_like(out)
        task_loss = criterion(out, target)
        # 稀疏正则
        mask_l1 = 0
        for m in masked_model.blocks:
            if hasattr(m, 'mask_logits'):
                for logit in m.mask_logits.values():
                    mask_l1 += torch.sigmoid(logit).mean()
        loss = task_loss + lambda_sparse * mask_l1
        loss.backward()
        optimizer.step()
        # 动态调整温度，初始1.0，每轮乘0.8
        new_temp = max(0.05, temperature * (0.8 ** epoch))
        for m in masked_model.blocks:
            if hasattr(m, 'update_temperature'):
                m.update_temperature(new_temp)
        print(f"Epoch {epoch+1}/{epochs} mask training done. Loss: {loss.item():.4f}, Temperature: {new_temp:.4f}")

    learngene_params = masked_model.get_masked_parameters()
    return learngene_params

def analyze_nonzero_parameters(model, print_details=False):
    total_nonzero = 0
    total_params = 0
    layer_stats = {}
    for name, param in model.named_parameters():
        numel = param.numel()
        nonzero = (param != 0).sum().item()
        layer_stats[name] = (nonzero, numel)
        total_nonzero += nonzero
        total_params += numel

    if print_details:
        print("\nNon-zero parameter distribution (sorted by block):")
        block_layers = []
        for name in layer_stats:
            if name.startswith("blocks."):
                parts = name.split('.')
                if len(parts) > 2 and parts[1].isdigit():
                    block_idx = int(parts[1])
                    subname = '.'.join(parts[2:])
                    block_layers.append((block_idx, subname, name))
        block_layers.sort()
        for block_idx, subname, name in block_layers:
            nonzero, numel = layer_stats[name]
            sparse = 100.0 * (1 - nonzero / numel)
            print(f"{name}: {nonzero}/{numel} ({sparse:.2f}% sparse)")
        for name in sorted(layer_stats.keys()):
            if not name.startswith("blocks."):
                nonzero, numel = layer_stats[name]
                sparse = 100.0 * (1 - nonzero / numel)
                print(f"{name}: {nonzero}/{numel} ({sparse:.2f}% sparse)")
    return total_nonzero, total_params, layer_stats

def main():
    parser = argparse.ArgumentParser('Learngene Extraction Example')
    parser.add_argument('--data_set', default='CIFAR10', type=str, help='Dataset name: CIFAR10 | CIFAR100 | IMNET')
    parser.add_argument('--data_path', default='./data', type=str, help='Path to dataset root')
    parser.add_argument('--model', default='deit_base_patch16_224', type=str)
    parser.add_argument('--num-blocks', default=12, type=int)
    parser.add_argument('--epochs', default=10, type=int)
    parser.add_argument('--lr', default=0.01, type=float)
    parser.add_argument('--temperature', default=1.0, type=float)
    parser.add_argument('--threshold', type=float, required=True, help='Threshold for binary mask generation (must be specified)')
    parser.add_argument('--device', default='cuda', type=str)
    parser.add_argument('--print_details', action='store_true', help='Print per-layer nonzero parameter stats')
    args = parser.parse_args()

    ancestor_model = VisionTransformer(depth=args.num_blocks, mask_training=False).to(args.device)
    print(f"Ancestor model parameter count: {sum(p.numel() for p in ancestor_model.parameters())}")

    learngene_params = get_masked_parameters(
        ancestor_model,
        num_blocks=args.num_blocks,
        epochs=args.epochs,
        lr=args.lr,
        temperature=args.temperature,
        threshold=args.threshold,
        device=args.device
    )

    aux_model = VisionTransformer(depth=args.num_blocks, mask_training=False).to(args.device)
    aux_model.load_state_dict(learngene_params, strict=False)
    print(f"Auxiliary model parameter count after training: {sum(p.numel() for p in aux_model.parameters())}")

    total_nonzero, total_params, layer_stats = analyze_nonzero_parameters(aux_model, args.print_details)
    print(f"\nTotal parameters in auxiliary model: {total_params}")
    print(f"Non-zero parameters in auxiliary model: {total_nonzero} ({100.0 * total_nonzero / total_params:.2f}%)")
    print("Learngene successfully extracted and loaded into auxiliary model")

if __name__ == '__main__':
    main()