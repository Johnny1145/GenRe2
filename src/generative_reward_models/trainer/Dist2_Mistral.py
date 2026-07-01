import os
import torch
from dataclasses import dataclass, field
from datasets import load_dataset
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import LoraConfig, TaskType, PeftModel, get_peft_model_state_dict
from trl import (
    ModelConfig,
    SFTConfig,
    SFTTrainer,
    TrlParser,
    get_kbit_device_map,
    get_quantization_config,
    get_peft_config,
)
from .loss import get_digit_loss
from collections import defaultdict
from typing import Dict, Literal

class Dist2LossTrainer(SFTTrainer):
    def __init__(self, *args, target_temperature=2.0, beta=1.0, **kwargs):
        super().__init__(*args, **kwargs)
        self._stored_metrics = defaultdict(lambda: defaultdict(list))
        self.target_temperature = target_temperature
        self.beta = beta

    def store_metrics(
        self, metrics: Dict[str, float], train_eval: Literal["train", "eval"] = "train"
    ) -> None:
        for key, value in metrics.items():
            self._stored_metrics[train_eval][key].append(value)

    def log(self, logs: Dict[str, float], *args, **kwargs) -> None:
        """
        Log `logs` on the various objects watching training, including stored metrics.
        """
        train_eval = "train" if "loss" in logs else "eval"
        for key, metrics in self._stored_metrics[train_eval].items():
            logs[key] = torch.tensor(metrics).mean().item()
        del self._stored_metrics[train_eval]
        return super().log(logs, *args, **kwargs)

    def compute_loss(self, model, inputs, return_outputs=False, **kwargs):
        labels = inputs.get("labels")
        outputs = model(**inputs)
        logits = outputs.get("logits")
        sft_loss = outputs.loss

        loss, stats = get_digit_loss(
            loss_sft=sft_loss,
            logits=logits,
            labels=labels,
            tokenizer=self.processing_class if hasattr(self, "processing_class") else self.tokenizer,
            target_temperature=self.target_temperature,
            beta=self.beta,
        )

        self.store_metrics(stats, train_eval="train")
        return (loss, outputs) if return_outputs else loss

# ---  1  SwanLab  ---
try:
    from swanlab.integration.huggingface import SwanLabCallback
except ImportError:
    raise ImportError(" swanlab: pip install swanlab")

#  WandB
os.environ["WANDB_DISABLED"] = "true"

# --- 1.  Table 4  Prompt  ---
INPUT_TEMPLATE = """###Task Description:
An instruction (might include an Input inside it), a response to evaluate, a reference answer that gets a score of 5, and a score rubric representing an evaluation criterion is given.
1. Write a detailed feedback that assesses the quality of the response strictly based on the given score rubric, not evaluating in general.
2. After writing a feedback, write a score that is an integer between 1 and 5. You should refer to the score rubric.
3. The output format should look as follows: Feedback: (write a feedback for criteria) [RESULT] (an integer number between 1 and 5)
4. Please do not generate any other opening, closing, and explanations.

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
    max_seq_length: int = field(
        default=2048,
        metadata={"help": "The maximum sequence length for SFT Trainer"}
    )
    target_temperature: float = field(
        default=2.0,
        metadata={"help": "Target temperature for digit loss"}
    )
    beta: float = field(
        default=1.0,
        metadata={"help": "Beta coefficient for digit loss"}
    )

if __name__ == "__main__":

    # 1.
    parser = TrlParser((ScriptArguments, SFTConfig, ModelConfig))
    script_args, training_args, model_config = parser.parse_args_into_dataclasses()

    #
    if training_args.output_dir == "tmp_trainer" or training_args.output_dir == ".":
        training_args.output_dir = os.path.abspath("./outputs/sft_Mistral_test")

    # ---  1  Appendix A  ---
    if training_args.learning_rate == 5e-5:
        training_args.learning_rate = 1.0e-5

    training_args.lr_scheduler_type = "cosine"
    training_args.save_total_limit = 2

    #  SFT
    training_args.dataset_text_field = "text"
    training_args.max_seq_length = script_args.max_seq_length

    #  SwanLab
    swanlab_callback = SwanLabCallback(
        project="Prometheus-Eval-SFT",  #
        experiment_name="llama3-8b-lora-sft", #
        description="Replicating Prometheus paper Table 4 settings with SwanLab visualization.",
        config={
            "model_name": model_config.model_name_or_path,
            "max_seq_length": script_args.max_seq_length,
            "lora_rank": 8,
            "learning_rate": training_args.learning_rate
        }
    )

    # 2.  Tokenizer
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

    model = AutoModelForCausalLM.from_pretrained(
        model_config.model_name_or_path, **model_kwargs
    )

    # ---  2 LoRA  (Appendix A) ---
    peft_config = LoraConfig(
        r=8,
        lora_alpha=16,
        lora_dropout=0.05,
        task_type=TaskType.CAUSAL_LM,
        target_modules="all-linear"
    )

    # 3.
    ds = load_dataset("prometheus-eval/Feedback-Collection")

    # 4.
    def format_for_sft(example):
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

        cot = example.get('orig_feedback', example.get('feedback', ""))
        raw_score = example.get('orig_score', example.get('score', 5))
        try:
            score = int(float(raw_score))
        except:
            score = 5

        phrase = "So the overall score is"

        if phrase in cot:
            assistant_content = cot
        else:
            assistant_content = f"{cot.strip()}\n{phrase} {score}"

        messages = [
            {"role": "user", "content": user_content},
            {"role": "assistant", "content": assistant_content}
        ]

        text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=False)
        return {"text": text}

    train_dataset = ds['train'].map(format_for_sft, num_proc=8)

    # 5.  Dist2LossTrainer
    trainer = Dist2LossTrainer(
        model=model,
        processing_class=tokenizer,
        args=training_args,
        train_dataset=train_dataset,
        peft_config=peft_config,
        # ---  3  callbacks ---
        callbacks=[swanlab_callback],
        target_temperature=script_args.target_temperature,
        beta=script_args.beta,
    )

    # 6.
    print("Training started with SwanLab tracking...")
    trainer.train()

    # 7.
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
