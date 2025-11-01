import time 
import heapq 
import torch 
import torch.nn as nn 
from .sparsegpt import SparseGPT 
from .layerwrapper import WrappedGPT
from .data import get_loaders 

from .ablate import AblateGPT 

def find_layers(module, layers=[nn.Linear], name=''):
    """
    Recursively find the layers of a certain type in a module.

    Args:
        module (nn.Module): PyTorch module.
        layers (list): List of layer types to find.
        name (str): Name of the module.

    Returns:
        dict: Dictionary of layers of the given type(s) within the module.
    """
    if type(module) in layers:
        return {name: module}
    res = {}
    for name1, child in module.named_children():
        res.update(find_layers(
            child, layers=layers, name=name + '.' + name1 if name != '' else name1
        ))
    return res

def check_sparsity(model):
    use_cache = model.config.use_cache 
    model.config.use_cache = False 

    layers = model.model.layers
    count = 0 
    total_params = 0
    for i in range(len(layers)):
        layer = layers[i]
        subset = find_layers(layer)

        sub_count = 0
        sub_params = 0
        for name in subset:
            W = subset[name].weight.data
            count += (W==0).sum().item()
            total_params += W.numel()

            sub_count += (W==0).sum().item()
            sub_params += W.numel()

        print(f"layer {i} sparsity {float(sub_count)/sub_params:.6f}")

    model.config.use_cache = use_cache 
    return float(count)/total_params 

def prepare_calibration_input(model, dataloader, device):
    use_cache = model.config.use_cache
    model.config.use_cache = False
    layers = model.model.layers

    # dev = model.hf_device_map["model.embed_tokens"]
    if "model.embed_tokens" in model.hf_device_map:
        device = model.hf_device_map["model.embed_tokens"]

    dtype = next(iter(model.parameters())).dtype
    inps = torch.zeros((128, model.seqlen, model.config.hidden_size), dtype=dtype, device=device)
    inps.requires_grad = False
    cache = {'i': 0, 'attention_mask': None, "position_ids": None}

    class Catcher(nn.Module):
        def __init__(self, module):
            super().__init__()
            self.module = module
        def forward(self, inp, **kwargs):
            inps[cache['i']] = inp
            cache['i'] += 1
            cache['attention_mask'] = kwargs['attention_mask']
            cache['position_ids'] = kwargs['position_ids']
            raise ValueError
    layers[0] = Catcher(layers[0])
    for batch in dataloader:
        try:
            model(batch[0].to(device))
        except ValueError:
            pass  
    layers[0] = layers[0].module

    outs = torch.zeros_like(inps)
    attention_mask = cache['attention_mask']
    print('attention_mask')
    print(attention_mask)
    position_ids = cache['position_ids']
    model.config.use_cache = use_cache

    return inps, outs, attention_mask, position_ids 

def return_given_alpha(alpha, sort_res, W_metric, tmp_metric, sum_before):
    thres_cumsum = sum_before * alpha 
    sort_mask = tmp_metric <= thres_cumsum.reshape((-1,1))
    thres = torch.gather(sort_res[0], dim=1, index=sort_mask.sum(dim=1, keepdims=True)-1)
    W_mask = (W_metric <= thres)
    cur_sparsity = (W_mask==True).sum() / W_mask.numel()
    return W_mask, cur_sparsity

def prune_magnitude(args, model, tokenizer, device=torch.device("cuda:0"), prune_n=0, prune_m=0):
    """
    Magnitude-based pruning.
    - Supports unstructured (layer-wise, row-wise) and n:m structured.
    - Handles sparsity_ratio given as 0~1 or 0~100 (percent).
    - Guards against 100% pruning (keeps at least 1 weight per row/layer).
    """
    # normalize sparsity ratio to [0, 1)
    ratio = float(args.sparsity_ratio)
    if ratio > 1.0:
        ratio = ratio / 100.0
    ratio = max(0.0, min(0.999, ratio))  # forbid 100%

    layers = model.model.layers

    for i in range(len(layers)):
        layer = layers[i]
        subset = find_layers(layer)  # {qualified_name: nn.Linear}

        for name, mod in subset.items():
            W = mod.weight.data
            dev = W.device
            W_metric = torch.abs(W)  # magnitude metric on the same device

            if prune_n != 0:
                # ---------- structured n:m ----------
                # For each row, in every contiguous block of size m, zero the smallest n entries.
                m = int(prune_m)
                n = int(prune_n)
                assert m > 0 and n >= 0 and n < m, "Require 0 <= n < m for n:m pruning"

                rows, cols = W_metric.shape
                W_mask = torch.zeros_like(W_metric, dtype=torch.bool, device=dev)

                # iterate blocks by step m to avoid overlapping
                for start in range(0, cols, m):
                    end = min(start + m, cols)
                    block = W_metric[:, start:end]                  # [rows, block_width]
                    bw = block.shape[1]
                    if bw <= 1 or n == 0:
                        continue
                    # never drop entire block: cap n at bw-1
                    k = min(n, bw - 1)
                    idx = torch.topk(block, k, dim=1, largest=False).indices  # [rows, k]
                    # scatter True into the mask within the block range
                    W_mask[:, start:end].scatter_(1, idx, True)

            elif "row" in str(args.prune_method):
                # ---------- unstructured row-wise ----------
                # For each row, zero the smallest k = int(cols * ratio) weights.
                rows, cols = W_metric.shape
                if cols == 0:
                    continue
                k = min(int(cols * ratio), max(cols - 1, 0))  # keep at least 1 per row
                if k <= 0:
                    # nothing to prune on this layer by ratio
                    W_mask = torch.zeros_like(W_metric, dtype=torch.bool, device=dev)
                else:
                    # get indices of k smallest per row
                    idx = torch.topk(W_metric, k, dim=1, largest=False).indices  # [rows, k]
                    W_mask = torch.zeros_like(W_metric, dtype=torch.bool, device=dev)
                    W_mask.scatter_(1, idx, True)

            else:
                # ---------- unstructured layer-wise ----------
                # Over the whole tensor, zero the smallest k = int(N * ratio) weights.
                N = W_metric.numel()
                if N == 0:
                    continue
                k = min(int(N * ratio), max(N - 1, 0))  # keep at least 1 weight overall
                if k <= 0:
                    W_mask = torch.zeros_like(W_metric, dtype=torch.bool, device=dev)
                else:
                    # kth smallest threshold via topk (largest=False) and take the max among the k selected
                    flat_vals = W_metric.view(-1)
                    vals, _ = torch.topk(flat_vals, k, largest=False)
                    thresh = vals.max()  # threshold among k smallest
                    W_mask = (W_metric <= thresh)

            # apply mask
            W[W_mask] = 0

def prune_wanda(args, model, tokenizer, device=torch.device("cuda:0"), prune_n=0, prune_m=0):
    # Disable KV cache to get full activations
    use_cache = model.config.use_cache
    model.config.use_cache = False

    print("loading calibration data")
    dataloader, _ = get_loaders("c4", nsamples=args.nsamples, seed=args.seed, seqlen=model.seqlen, tokenizer=tokenizer)
    print("dataset loading complete")

    layers = model.model.layers

    # 1) Wrap target Linear layers once, and register hooks (end-to-end path)
    wrapped_layers = {}
    handles = []

    def find_and_wrap(layer_module, prefix):
        subset = find_layers(layer_module)  # {qualified_name: nn.Linear}
        for name, mod in subset.items():
            qname = f"{prefix}.{name}"
            wrapped_layers[qname] = WrappedGPT(mod)

            # forward hook to collect per-batch inputs/outputs
            def add_batch(name_in_hook):
                def _hook(_module, inp, out):
                    # inp: tuple(tensor,), we need [B, T, H]
                    x = inp[0].data
                    y = out.data
                    wrapped_layers[name_in_hook].add_batch(x, y)
                return _hook

            handles.append(mod.register_forward_hook(add_batch(qname)))

    for i, layer in enumerate(layers):
        find_and_wrap(layer, f"model.layers.{i}")

    # 2) Single end-to-end forward passes to populate statistics
    #    DO NOT call each layer directly; call the whole model.
    with torch.no_grad():
        ncollected = 0
        for batch in dataloader:
            # batch[0]: input_ids; this path builds proper masks/position_ids internally
            input_ids = batch[0].to(device)
            _ = model(input_ids)  # full-model forward; RoPE, masks are handled internally
            ncollected += 1
            if ncollected >= args.nsamples:
                break

    # 3) Pruning per wrapped Linear (Wanda)
    for qname, wrapper in wrapped_layers.items():
        mod = wrapper.module  # the actual nn.Linear
        W = mod.weight.data
        W_metric = torch.abs(W) * torch.sqrt(wrapper.scaler_row.reshape((1, -1)))

        if prune_n != 0:
            # structured n:m
            W_mask = (torch.zeros_like(W_metric) == 1)
            m = prune_m
            for ii in range(W_metric.shape[1]):
                if ii % m == 0:
                    tmp = W_metric[:, ii:(ii+m)].float()
                    idx = torch.topk(tmp, prune_n, dim=1, largest=False)[1]
                    W_mask.scatter_(1, ii + idx, True)
        else:
            # unstructured Wanda (per-row sort)
            sort_res = torch.sort(W_metric, dim=-1, stable=True)
            if args.use_variant:
                tmp_metric = torch.cumsum(sort_res[0], dim=1)
                sum_before = W_metric.sum(dim=1)
                alpha = 0.4
                alpha_hist = [0., 0.8]
                W_mask, cur_sparsity = return_given_alpha(alpha, sort_res, W_metric, tmp_metric, sum_before)
                while (torch.abs(cur_sparsity - args.sparsity_ratio) > 0.001) and (alpha_hist[1] - alpha_hist[0] >= 0.001):
                    if cur_sparsity > args.sparsity_ratio:
                        alpha_new = (alpha + alpha_hist[0]) / 2.0
                        alpha_hist[1] = alpha
                    else:
                        alpha_new = (alpha + alpha_hist[1]) / 2.0
                        alpha_hist[0] = alpha
                    alpha = alpha_new
                    W_mask, cur_sparsity = return_given_alpha(alpha, sort_res, W_metric, tmp_metric, sum_before)
                print(f"[{qname}] alpha={alpha:.4f}, sparsity={cur_sparsity:.6f}")
            else:
                # plain unstructured: per-row top-k to zero
                k = int(W_metric.shape[1] * args.sparsity_ratio)
                idx = sort_res[1][:, :k]
                W_mask = (torch.zeros_like(W_metric) == 1)
                W_mask.scatter_(1, idx, True)

        W[W_mask] = 0  # apply mask

    # 4) Cleanup hooks
    for h in handles:
        h.remove()

    # 5) (Optional) Sanity forward to ensure model runs fine post-pruning
    with torch.no_grad():
        _ = model(batch[0].to(device))

    model.config.use_cache = use_cache
    torch.cuda.empty_cache()
@torch.no_grad()
def prune_sparsegpt(args, model, tokenizer, dev, prune_n=0, prune_m=0):
    """
    SparseGPT pruning using full-model forwards + forward hooks only.
    - No per-layer direct forward calls (avoids RoPE/position_embeddings issues).
    - Collect input/output stats via hooks, then run fasterprune().
    """
    # --- config ---
    print("Starting ...")
    dataloader, _ = get_loaders("c4", nsamples=args.nsamples, seed=args.seed,
                                seqlen=model.seqlen, tokenizer=tokenizer)

    use_cache = model.config.use_cache
    model.config.use_cache = False
    layers = model.model.layers

    # Infer device for inputs from embed_tokens if available
    if "model.embed_tokens" in getattr(model, "hf_device_map", {}):
        dev = model.hf_device_map["model.embed_tokens"]

    print("Ready.")

    # parse sparsegpt args
    blockdim = 0 if ("row" in args.prune_method) else 1
    print(f"blockdim {blockdim}")

    # normalize sparsity ratio to [0, 1); forbid 100% pruning
    ratio = float(args.sparsity_ratio)
    if ratio > 1.0:
        ratio = ratio / 100.0
    ratio = max(0.0, min(0.999, ratio))

    # --- 1) register hooks for all target Linear modules in all layers ---
    gpts = {}      # qname -> SparseGPT
    modules = {}   # qname -> module (nn.Linear), convenience
    handles = []

    def find_and_prepare(layer_module, prefix):
        subset = find_layers(layer_module)  # {qualified_name: nn.Linear}
        for name, mod in subset.items():
            qname = f"{prefix}.{name}"
            modules[qname] = mod
            gpts[qname] = SparseGPT(mod, blockdim=blockdim)

            # hook to accumulate stats inside SparseGPT
            def add_batch(name_in_hook):
                def _hook(_module, inp, out):
                    # inp is a tuple -> inp[0] is the tensor we need
                    x = inp[0].data
                    y = out.data
                    gpts[name_in_hook].add_batch(x, y)
                return _hook

            handles.append(mod.register_forward_hook(add_batch(qname)))

    for i, layer in enumerate(layers):
        find_and_prepare(layer, f"model.layers.{i}")

    # --- 2) run end-to-end forwards to populate stats (no per-layer calls) ---
    collected = 0
    for batch in dataloader:
        input_ids = batch[0].to(dev)
        _ = model(input_ids)  # RoPE/masks handled internally by the model
        collected += 1
        if collected >= args.nsamples:
            break

    # remove hooks before pruning
    for h in handles:
        h.remove()

    # --- 3) run SparseGPT.fasterprune() on each collected module ---
    # parse blocksize from prune_method (e.g., "gpt-row-128"), default 128
    toks = str(args.prune_method).split("-")
    blocksize = int(toks[2]) if len(toks) > 2 and toks[2].isdigit() else 128
    print(f"blocksize {blocksize}")

    for qname, g in gpts.items():
        print(f"Pruning {qname} ...")
        g.fasterprune(ratio, prune_n=prune_n, prune_m=prune_m,
                      percdamp=0.01, blocksize=blocksize)
        g.free()

    # --- 4) optional sanity forward post-pruning ---
    try:
        _ = model(input_ids)
    except Exception as e:
        print(f"[warn] post-prune forward check failed: {e}")

    # --- cleanup ---
    model.config.use_cache = use_cache
    torch.cuda.empty_cache()

@torch.no_grad()
def prune_ablate(args, model, tokenizer, dev, prune_n=0, prune_m=0):
    ## SparseGPT code available at: https://github.com/IST-DASLab/sparsegpt/tree/f5c25005a61f96a0933ca2f95705a963585aafaa
    print('Starting ...')
    dataloader, _ = get_loaders("c4",nsamples=args.nsamples,seed=args.seed,seqlen=model.seqlen,tokenizer=tokenizer)

    use_cache = model.config.use_cache
    model.config.use_cache = False
    layers = model.model.layers

    if "model.embed_tokens" in model.hf_device_map:
        dev = model.hf_device_map["model.embed_tokens"]

    dtype = next(iter(model.parameters())).dtype
    inps = torch.zeros(
        (args.nsamples, model.seqlen, model.config.hidden_size), dtype=dtype, device=dev
    )
    cache = {'i': 0, 'attention_mask': None, "position_ids": None}

    class Catcher(nn.Module):
        def __init__(self, module):
            super().__init__()
            self.module = module
        def forward(self, inp, **kwargs):
            inps[cache['i']] = inp
            cache['i'] += 1
            cache['attention_mask'] = kwargs['attention_mask']
            cache['position_ids'] = kwargs['position_ids']
            raise ValueError
    layers[0] = Catcher(layers[0])
    for batch in dataloader:
        try:
            model(batch[0].to(dev))
        except ValueError:
            pass
    layers[0] = layers[0].module
    torch.cuda.empty_cache()

    outs = torch.zeros_like(inps)
    attention_mask = cache['attention_mask']
    position_ids = cache['position_ids']

    print('Ready.')

    for i in range(len(layers)):
        layer = layers[i]
        if f"model.layers.{i}" in model.hf_device_map:
            dev = model.hf_device_map[f"model.layers.{i}"]
            print(f"layer {i} device {dev}")
            inps, outs, attention_mask, position_ids = inps.to(dev), outs.to(dev), attention_mask.to(dev), position_ids.to(dev)

        subset = find_layers(layer)

        gpts = {}
        for name in subset:
            gpts[name] = AblateGPT(subset[name])

        def add_batch(name):
            def tmp(_, inp, out):
                gpts[name].add_batch(inp[0].data, out.data)
            return tmp

        handles = []
        for name in gpts:
            handles.append(subset[name].register_forward_hook(add_batch(name)))

        for j in range(args.nsamples):
            outs[j] = layer(inps[j].unsqueeze(0), attention_mask=attention_mask, position_ids=position_ids)[0]
        for h in handles:
            h.remove()

        for name in gpts:
            print(i, name)
            print('Pruning ...')

            if args.prune_method == "ablate_wanda_seq":
                prune_mask = gpts[name].get_wanda_mask(args.sparsity_ratio, prune_n, prune_m)
            elif args.prune_method == "ablate_mag_seq":
                prune_mask = gpts[name].get_mag_mask(args.sparsity_ratio, prune_n, prune_m)
            elif "iter" in args.prune_method:
                prune_mask = None 

            gpts[name].fasterprune(args, args.sparsity_ratio, mask=prune_mask, prune_n=prune_n, prune_m=prune_m, percdamp=0.01, blocksize=128)
            gpts[name].free()

        for j in range(args.nsamples):
            outs[j] = layer(inps[j].unsqueeze(0), attention_mask=attention_mask, position_ids=position_ids)[0]

        layers[i] = layer 
        torch.cuda.empty_cache()

        inps, outs = outs, inps

    model.config.use_cache = use_cache
    torch.cuda.empty_cache()