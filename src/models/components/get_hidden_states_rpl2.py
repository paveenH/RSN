#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Mar 10 15:58:52 2025

@author: paveenhuang
"""

#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Script to:
1) Load "None" and "Expert" hidden states from .npy files (per-sample).
2) For each sample, compute diff = expert - none (shape = [num_layers, hidden_size]).
3) 将不在 [start:end) 范围内的 layer-diff 置为 0。
4) 利用 _apply_diff_hooks 注入这些 diff，抽取最后一个 token 的 hidden states (pos1)。
5) 保存结果到新的 .npy 文件。

@author: ...
"""

import os
import argparse
import json
import numpy as np
from tqdm import tqdm
from vicuna import VicundaModel


def parse_arguments():
    parser = argparse.ArgumentParser(description="Inject per-sample diff via _apply_diff_hooks and extract final HS.")
    parser.add_argument("task", type=str, help="The name of the task to process.")
    parser.add_argument("size", type=str, help="Model size (e.g. '13B').")
    parser.add_argument("model", type=str, help="Model type (e.g. 'llama3').")
    parser.add_argument("--start", type=int, default=0, help="Start layer index for injecting diff (inclusive).")
    parser.add_argument("--end", type=int, default=1, help="End layer index (exclusive).")
    return parser.parse_args()


def main():
    args = parse_arguments()
    task = args.task
    size = args.size
    model_name = args.model

    start_layer = args.start
    end_layer = args.end

    # 1) Load model
    model_path = f"/data2/paveen/RolePlaying/shared/{model_name}/{size}"
    vc = VicundaModel(model_path=model_path)
    template = vc.template
    print(f"Loaded model: {model_name}, size={size}")
    print(f"Template:\n{template}")

    # 2) Prepare data
    mmlu_path = "/data2/paveen/RolePlaying/src/models/components/mmlu"
    json_path = os.path.join(mmlu_path, f"{task}.json")
    if not os.path.isfile(json_path):
        raise FileNotFoundError(f"JSON file '{json_path}' not found.")

    with open(json_path, "r", encoding="utf-8") as f:
        data = json.load(f)
    print(f"Loaded {len(data)} samples from {json_path}")

    # 3) Load original none & expert hidden states
    #    这些 hidden states 的 shape 约定是 (num_samples, 1, total_layers, hidden_size)
    #    其中 total_layers 包括 embedding layer (index=0)，一般要去掉它
    hs_dir = f"/data2/paveen/RolePlaying/src/models/components/hidden_states_v3/{model_name}"
    none_hs_path = os.path.join(hs_dir, f"none_{task}_{task}_{size}.npy")
    expert_hs_path = os.path.join(hs_dir, f"{task}_{task}_{size}.npy")

    if not (os.path.isfile(none_hs_path) and os.path.isfile(expert_hs_path)):
        raise FileNotFoundError(f"Cannot find HS npy files: {none_hs_path} or {expert_hs_path}")

    none_array = np.load(none_hs_path)   # shape: (num_samples, 1, total_layers, hidden_size)
    expert_array = np.load(expert_hs_path)
    if none_array.shape != expert_array.shape:
        raise ValueError("None & Expert hidden states shape mismatch.")

    # 去掉 embedding layer (index=0)，只保留后续 decoder layers
    none_array = none_array[:, :, 1:, :]    # => (num_samples, 1, num_layers, hidden_size)
    expert_array = expert_array[:, :, 1:, :]

    num_samples, _, num_layers, hidden_size = none_array.shape
    print(f"After removing embedding layer => shape: {none_array.shape}")
    print(f"  #samples={num_samples}, #layers={num_layers}, hidden_size={hidden_size}")

    # 检查层索引是否合法
    if not (0 <= start_layer < end_layer <= num_layers):
        raise ValueError(f"Invalid layer range: [start={start_layer}, end={end_layer}), "
                         f"must be in [0, {num_layers})")

    # 4) 存储最终注入差值后得到的 hidden states
    #    准备一个 list，最后 stack 成 numpy 数组
    diffed_hs_list = []

    # 5) Iterate each sample
    print(f"Injecting diff into layers [{start_layer}:{end_layer}), extracting final HS...")
    for idx, sample in enumerate(tqdm(data, desc="Processing Samples")):
        context = sample.get("text", "")
        if not context:
            continue

        if idx >= num_samples:
            print(f"Sample idx={idx} out of range for HS array (size={num_samples}). Stop.")
            break

        # 构造 "none" prompt
        none_character = f"none {task.replace('_',' ')}"
        prompt = template.format(character=none_character, context=context)

        # 该 sample 的 none / expert hidden states，shape = (num_layers, hidden_size)
        none_hs = none_array[idx, 0]       # [num_layers, hidden_size]
        expert_hs = expert_array[idx, 0]   # 同上

        # 计算差值 diff
        diff_matrix = expert_hs - none_hs  # [num_layers, hidden_size]

        # 如果只想在 [start_layer, end_layer) 注入 diff，则把其它层的 diff 置 0
        if start_layer > 0:
            diff_matrix[:start_layer, :] = 0
        if end_layer < num_layers:
            diff_matrix[end_layer:, :] = 0

        # 6) 调用 get_hidden_states_mdf 或自己包装
        #    这里直接用 get_hidden_states_mdf，其内部就是 _apply_diff_hooks + forward pass
        #    它返回 [pos1_hs]，其中 pos1_hs 是 shape=(num_layers, hidden_size)
        hs_modified_list = vc.get_hidden_states_mdf(prompt=prompt, diff_matrices=diff_matrix)

        # 通常我们只取 hs_modified_list[0] 这个位置
        if not hs_modified_list or hs_modified_list[0] is None:
            # 说明没有取到正确结果，跳过
            continue

        final_hs = hs_modified_list[0]  # shape = (num_layers, hidden_size)
        final_hs = np.expand_dims(final_hs, axis=0)  # => (1, num_layers, hidden_size)

        diffed_hs_list.append(final_hs)

    # 7) 统一堆叠并保存
    if not diffed_hs_list:
        print("No diffed hidden states were collected.")
        return

    diffed_arr = np.stack(diffed_hs_list, axis=0)
    # => shape: (num_samples, 1, num_layers, hidden_size)

    save_dir = f"/data2/paveen/RolePlaying/src/models/components/hidden_states_diff/{model_name}"
    os.makedirs(save_dir, exist_ok=True)

    out_file = os.path.join(save_dir, f"{task}_{size}_{start_layer}_{end_layer}.npy")
    np.save(out_file, diffed_arr)
    print(f"Saved diffed hidden states to: {out_file}")
    print("All done!")


if __name__ == "__main__":
    main()