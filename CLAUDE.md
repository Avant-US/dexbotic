# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

Dexbotic is a Vision-Language-Action (VLA) development toolbox for embodied intelligence research. It provides a unified framework for training, fine-tuning, and deploying VLA models (CogACT, Pi0/Pi0.5, OFT, MemVLA, DM0, GR00TN1, NaVILA, etc.) on robotic manipulation and navigation tasks.

- **Language**: Python 3.8+ / PyTorch
- **Version**: 0.2.0

## Build & Install

```bash
# Docker (recommended)
docker run -it --rm --gpus all --network host -v /path/to/dexbotic:/dexbotic dexmal/dexbotic bash
cd /dexbotic && conda activate dexbotic && pip install -e .

# Local
conda create -n dexbotic python=3.10 -y && conda activate dexbotic
pip install torch==2.2.2 torchvision==0.17.2 xformers --index-url https://download.pytorch.org/whl/cu118
pip install -e .
pip install transformers==4.51.0 ninja packaging
pip install flash-attn --no-build-isolation  # optional
```

## Common Commands

```bash
# Training (8 GPUs recommended)
torchrun --nproc_per_node=8 playground/benchmarks/libero/libero_cogact.py

# Training with DeepSpeed zero3 offload (for smaller GPUs like RTX 4090)
torchrun --nproc_per_node=8 --nnodes=1 --node_rank=0 --master_addr=localhost --master_port=29500 \
  playground/benchmarks/libero/libero_cogact.py --deepspeed script/deepspeed/zero3_offload.json

# Inference server
CUDA_VISIBLE_DEVICES=0 python playground/benchmarks/libero/libero_cogact.py --task inference

# Single inference
CUDA_VISIBLE_DEVICES=0 python playground/benchmarks/libero/libero_cogact.py --task inference_single \
  --image_path test_data/libero_test.png --prompt 'What action should the robot take?'

# Data conversion
python script/convert_data/convert_lerobot_to_dexdata.py
python script/convert_data/convert_rlds_to_dexdata.py

# Formatting
autopep8 --in-place --aggressive --aggressive --recursive .
black dexbotic/
isort dexbotic/
```

## Architecture

The framework has three layers: **Data**, **Model**, and **Experiment**.

### Data Layer (`dexbotic/data/`)

- **Dexdata format**: Episodes stored as JSONL files with images/video, actions, prompts, and states.
- **Dataset registry** (`data/data_source/`): Datasets registered via `register_dataset()` into the global `CONVERSATION_DATA` dict. Modules are auto-imported at package load. External datasets loaded via `DEXBOTIC_DATA_PATH` env var.
- **DexDataset** (`data/dataset/dex_dataset.py`): Unified dataset class accepting tokenization, action processing, and image processing callables.
- **Transform pipeline** (`data/dataset/transform/`): Composable `Pipeline` of transforms split by domain — `action.py` (normalization, delta, padding), `language.py` (prompt templates, conversations), `multimodal.py` (image/video loading), `common.py` (format conversions).
- **Collator** (`data/collator.py`): `DataCollatorForSupervisedDataset` pads sequences and stacks tensors for batching.

### Model Layer (`dexbotic/model/`)

- **Base classes** (`dexbotic_arch.py`): `DexboticConfig`, `DexboticVLMModel` (LLM + vision tower + projector), `DexboticForCausalLM` (adds LM head).
- **Model-specific subdirs**: `cogact/`, `pi0/`, `pi05/`, `dm0/`, `gr00tn1/`, `memvla/`, `oft/`, `navila/`, `uninavid/`, `muvla/`, `discrete_vla/`. Each follows the pattern: Config → Model → ForCausalLM.
- **Vision modules** (`model/modules/mm_vision/`): Factory `build_vision_tower()` supporting CLIP, SigLIP, EVA-ViT encoders.
- **Projector modules** (`model/modules/mm_projector/`): Factory `build_vision_projector()` with linear, MLP, and downsampling options.
- **Action heads**: Model-specific (diffusion/DiT for CogACT/MemVLA, flow-matching for DM0, linear for Pi0).
- Models register with HuggingFace `AutoModel`/`AutoConfig` for `from_pretrained()` discovery.

### Experiment Layer (`dexbotic/exp/`)

- **BaseExp** (`base_exp.py`): Orchestrates training and inference. Composed of dataclass configs: `OptimizerConfig`, `TrainerConfig`, `ModelConfig`, `DataConfig`, `ActionConfig`, `InferenceConfig`.
- **Model-specific exps** (`cogact_exp.py`, `pi0_exp.py`, `dm0_exp.py`, etc.): Override config defaults for each model type.
- **DexboticTrainer** (`trainer.py`): Extends HuggingFace `Trainer` with per-module learning rates, custom schedulers, ZeRO-3 compatible saving.
- **Benchmark scripts** (`playground/benchmarks/`): Entry points that subclass the model experiment, override configs, and dispatch `exp.train()` or `exp.inference()`.

### Experiment Configuration Pattern

All experiments follow this pattern — create a benchmark script in `playground/` that subclasses a model's Exp and overrides config dataclass fields:

```python
@dataclass
class MyTrainerConfig(CogACTTrainerConfig):
    output_dir: str = "./user_checkpoints/my_exp"
    num_train_epochs: int = 25

@dataclass
class MyExp(CogACTExp):
    trainer_config: MyTrainerConfig = field(default_factory=MyTrainerConfig)

if __name__ == "__main__":
    exp = MyExp()
    exp.train()  # or exp.inference()
```

See `playground/example_exp.py` for the canonical template.

### Adding a New Model

Extend three classes in a new subdir under `dexbotic/model/`:
1. `MyConfig(DexboticConfig)` — custom parameters (action_dim, chunk_size, etc.)
2. `MyModel(DexboticVLMModel)` — `_build_*_module()` methods, `initialize_model()`
3. `MyForCausalLM(DexboticForCausalLM)` — `forward()` returning `CausalLMOutputDexbotic`

Register with `AutoConfig.register()` / `AutoModel.register()` in `__init__.py`. Then create a matching experiment class in `dexbotic/exp/`. See `docs/web_docs/8. Develop Your Own Model.md`.

### Key Constants (`dexbotic/constants.py`)

`IGNORE_INDEX = -100`, `IMAGE_TOKEN_INDEX = -200`, `DEFAULT_IMAGE_TOKEN = "<image>"`

### Inference / Deployment

- Server: Flask-based, launched via `exp.inference()` on a configurable port (default 7891).
- Client: `DexClient` (`dexbotic/client.py`) sends images + prompts over HTTP, receives action sequences. Handles delta-to-absolute conversion and action queue smoothing.

## Key Dependencies

torch, transformers, peft, bitsandbytes, timm, diffusers, megfile, loguru, wandb, flash-attn (optional), deepspeed (for distributed training)

## PR Checklist

PRs must confirm: no breaking changes to VLA API / Dexdata format, no redundant implementations, code follows existing design patterns, benchmark evaluation attached, no regressions. See `.github/pull_request_template.md`.

## 你是谁

+ 你是python领域的专家
+ 你是RL领域的专家
+ 你是机器人领域的专家
+ 你是LLM, VLM, VLA, WAM领域的专家

## 必须参考

+ 本 dexbotic 代码库来自 https://github.com/dexmal/dexbotic , 可参考该github库上面的 Issues, Pull Requests, Discussion 等等. 
+ dexbotic 对应的官方文档在 https://dexbotic.com/docs/ . 除了官方文档外, 你还可以参考本库 @docs/ 内的所有pdf和md文档, 也可参考 @b/d/ 内的所有pdf和md文档.
+ 也可参考 @b/d/dm0/ 内的所有DM0模型相关的pdf和md文档. 而 @b/m/dm0/ 内保存的是不同版本的 DM0 模型 checkpoint 的词表与模型配置文件. 比如, 在 @b/m/dm0/base/ 中保存的是模型 https://huggingface.co/Dexmal/DM0-base 也就是DM0的base版本的配置文件. 
+ 除了参考本地文档和官网外, 也可搜索网上其它相关的文章, 
+ 但一定要参考 dexbotic 本地代码的现实情况.