import os
import re
import torch
from dataclasses import dataclass, field
from datasets import load_dataset
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import LoraConfig, TaskType
from trl import (
    ModelConfig,
    GRPOConfig,
    ReMaxTrainer,
    TrlParser,
    get_kbit_device_map,
    get_quantization_config,
    get_peft_config,
)

# --- SwanLab  ---
try:
    from swanlab.integration.huggingface import SwanLabCallback
except ImportError:
    raise ImportError(" swanlab: pip install swanlab")

os.environ["WANDB_DISABLED"] = "true"

# --- 1.  Prompt  ---
#  Prompt
INPUT_TEMPLATE = """###Task Description:
An instruction (might include an Input inside it), a response to evaluate, a reference answer that gets a score of 5, and a score rubric representing an evaluation criterion is given.
1. Write a detailed feedback that assesses the quality of the response strictly based on the given score rubric.
2. After writing a feedback, write a score that is an integer between 1 and 5.
3. The output MUST end with the exact phrase: "So the overall score is [score]" where [score] is an integer 1-5.

###The instruction to evaluate:
{orig_instruction}

###Response to evaluate:
{orig_response}

###Reference Answer (Score 5):
{orig_reference_answer}

###Score Rubrics:
[{orig_criteria}]
Score 1: {orig_score1_description}
Score 2: {orig_score2_description}
Score 3: {orig_score3_description}
Score 4: {orig_score4_description}
Score 5: {orig_score5_description}

###Feedback:
"""

@dataclass
class ScriptArguments:
    max_seq_length: int = field(default=2048, metadata={"help": "Max input length"})
    ref_model_name_or_path: str = field(
        default=None,
        metadata={"help": "Optional: Path to the reference model for KL penalty. If None, uses the base model with disabled adapters."}
    )

# ---   ---
# TRL
#  'ground_truth_score'
def accuracy_and_format_reward_func(prompts, completions, **kwargs):
    rewards = []
    #  "So the overall score is <>"
    pattern = r"So the overall score is\s+(\d+)"

    ground_truth_scores = kwargs.get("ground_truth_score", [])

    for i, completion in enumerate(completions):
        # --- 1.  ( Chat ) ---
        text_content = ""

        #  A: completion  (: [{'role': 'assistant', 'content': '...'}])
        if isinstance(completion, list):
            #
            if len(completion) > 0 and isinstance(completion[-1], dict):
                text_content = completion[-1].get("content", "")
        #  B: completion  ()
        elif isinstance(completion, str):
            text_content = completion
        else:
            text_content = str(completion)

        # --- 2.  ---
        reward = 0.0
        pred_score = None

        #  text_content
        matches = list(re.finditer(pattern, text_content, re.IGNORECASE))
        if matches:
            last_match = matches[-1]
            try:
                pred_score = int(last_match.group(1))
                #
                reward += 1.0
            except ValueError:
                pass

        if pred_score is None:
            #
            reward = -20.0
        else:
            #
            if i < len(ground_truth_scores):
                try:
                    gt_score = int(ground_truth_scores[i])
                    diff = abs(pred_score - gt_score) ** 2
                    # : 1.0 -
                    reward -= diff
                except Exception:
                    pass

        rewards.append(reward)

    return rewards

if __name__ == "__main__":
    parser = TrlParser((ScriptArguments, GRPOConfig, ModelConfig))
    script_args, training_args, model_config = parser.parse_args_into_dataclasses()

    #
    if not training_args.output_dir or training_args.output_dir == "tmp_trainer" or training_args.output_dir == ".":
        training_args.output_dir = os.path.abspath("./outputs/rl_ReMax_Mistral")

    # ---  ---
    if training_args.learning_rate == 5e-5:
        training_args.learning_rate = 1.0e-5

    training_args.max_completion_length = 512

    #  Gradient Checkpointing
    #  ZeRO-3 + GRPO  IndexError: pop from an empty deque
    # if training_args.gradient_checkpointing:
    #     training_args.gradient_checkpointing_kwargs = {"use_reentrant": False}

    # swanlab_callback = SwanLabCallback(
    #     project="Prometheus-Eval-ReMax",
    #     experiment_name="llama3-8b-ReMax-accuracy",
    #     description="GRPO with combined format penalty (-5) and accuracy reward (-|diff|).",
    #     config={
    #         "model_name": model_config.model_name_or_path,
    #         "reward_function": "Format(-5) + Accuracy(-|diff|)",
    #         "num_generations": training_args.num_generations,
    #     }
    # )

    # ---  Tokenizer ---
    torch_dtype = (
        model_config.torch_dtype
        if model_config.torch_dtype in ["auto", None]
        else getattr(torch, model_config.torch_dtype)
    )
    quantization_config = get_quantization_config(model_config)

    model_kwargs = dict(
        revision=model_config.model_revision,
        trust_remote_code=model_config.trust_remote_code,
        attn_implementation=model_config.attn_implementation,
        torch_dtype=torch_dtype,
        # ZeRO-3  device_map
        use_cache=False if training_args.gradient_checkpointing else True,
        device_map=get_kbit_device_map() if quantization_config is not None else None,
        quantization_config=quantization_config,
    )

    tokenizer = AutoTokenizer.from_pretrained(
        model_config.model_name_or_path,
        trust_remote_code=model_config.trust_remote_code,
        use_fast=True
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "left" # GRPO

    # ---  ---
    #  SFT  Base
    print(f"Loading model from {model_config.model_name_or_path}...")
    model = AutoModelForCausalLM.from_pretrained(
        model_config.model_name_or_path, **model_kwargs
    )

    #  LoRA
    model.enable_input_require_grads()
    model.config.use_cache = False

    peft_config = LoraConfig(
        r=8, #  rank
        lora_alpha=16,
        lora_dropout=0.05,
        task_type=TaskType.CAUSAL_LM,
        target_modules="all-linear"
    )

    # ---  ---
    ds = load_dataset("prometheus-eval/Feedback-Collection")

    def format_for_grpo(example):
        input_kwargs = {
            "orig_instruction": example.get('orig_instruction', example.get('instruction', '')),
            "orig_response": example.get('orig_response', example.get('output', example.get('response', ''))),
            "orig_reference_answer": example.get('orig_reference_answer', example.get('reference_answer', "N/A")),
            "orig_criteria": example.get('orig_criteria', example.get('rubric', "")),
            "orig_score1_description": example.get('orig_score1_description', "N/A"),
            "orig_score2_description": example.get('orig_score2_description', "N/A"),
            "orig_score3_description": example.get('orig_score3_description', "N/A"),
            "orig_score4_description": example.get('orig_score4_description', "N/A"),
            "orig_score5_description": example.get('orig_score5_description', "N/A"),
        }

        user_content = INPUT_TEMPLATE.format(**input_kwargs)
        if not user_content.strip():
            user_content = "Evaluate the quality of the response." #  Prompt

        messages = [{"role": "user", "content": user_content}]

        #
        raw_score = example.get('orig_score', example.get('score', 5))
        try:
            score = int(float(raw_score))
        except:
            score = 5 #

        return {
            "prompt": messages,
            # GRPOTrainer
            "ground_truth_score": score
        }

    train_dataset = ds['train'].map(format_for_grpo, num_proc=4)

    # ---  Prompt  () ---
    # if training_args.local_rank <= 0:
    #     print("\n" + "="*40 + " TRAIN PROMPT EXAMPLE " + "="*40)
    #     example_prompt = train_dataset[0]["prompt"]
    #     if isinstance(example_prompt, list):
    #         #  Chat
    #         print(example_prompt[-1]["content"])
    #     else:
    #         print(example_prompt)
    #     print("="*102 + "\n")
    # assert 0

    # ---  ---
    trainer = ReMaxTrainer(
        model=model,
        processing_class=tokenizer,
        #
        reward_funcs=[accuracy_and_format_reward_func],
        args=training_args,
        train_dataset=train_dataset,
        peft_config=peft_config,
        # callbacks=[swanlab_callback]
    )

    # ---  ---
    if training_args.beta > 0 and script_args.ref_model_name_or_path:
        from transformers import BitsAndBytesConfig
        from transformers.integrations.deepspeed import is_deepspeed_zero3_enabled

        print(f"Loading reference model: {script_args.ref_model_name_or_path}")

        # 1. 4-bit
        ref_quant_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_compute_dtype=torch.bfloat16,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_use_double_quant=True,
        )

        # 2.  ZeRO-3 ZeRO-3
        is_zero3 = is_deepspeed_zero3_enabled()

        load_kwargs = {
            "torch_dtype": torch.bfloat16,
            "trust_remote_code": model_config.trust_remote_code,
            "attn_implementation": model_config.attn_implementation,
            "low_cpu_mem_usage": True,
        }

        if not is_zero3:
            load_kwargs["quantization_config"] = ref_quant_config
            #
            load_kwargs["device_map"] = {"": trainer.accelerator.process_index}

        # 3.
        ref_model = AutoModelForCausalLM.from_pretrained(
            script_args.ref_model_name_or_path,
            **load_kwargs
        )

        # 4.  eval
        ref_model.eval()
        for param in ref_model.parameters():
            param.requires_grad = False

        trainer.ref_model = trainer.accelerator.prepare_model(ref_model, evaluation_mode=True)
        print(f"Reference model loaded. ZeRO-3 Mode: {is_zero3}, 4-bit Quantization: {not is_zero3}")

    print(f"Starting GRPO training with Beta={training_args.beta}, Format Penalty (-5) and Accuracy Reward...")
    trainer.train()

    # ---  ---
    trainer.accelerator.wait_for_everyone()
    model = trainer.accelerator.unwrap_model(trainer.model)

    #  PEFT  adapter
    #  ZeRO-3
    state_dict = trainer.model.state_dict()

    if trainer.is_world_process_zero:
        print(f"Training completed. Saving model to {training_args.output_dir}...")
        os.makedirs(training_args.output_dir, exist_ok=True)

        #  PEFT  state_dict
        #  model (unwrapped)  trainer.model
        model.save_pretrained(training_args.output_dir, state_dict=state_dict)

        tokenizer.save_pretrained(training_args.output_dir)
        print(f"Model and tokenizer saved successfully in {training_args.output_dir}")
