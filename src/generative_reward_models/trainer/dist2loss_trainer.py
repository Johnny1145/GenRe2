from trl import SFTTrainer
from .loss import get_digit_loss, get_digit_base_loss
from typing import Any, Callable, Dict, List, Literal, Optional, Tuple, Union
from collections import defaultdict
from ..prompt import *
import torch
import copy
torch.backends.cuda.matmul.allow_tf32 = True

class Dist2LossTrainer(SFTTrainer):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._stored_metrics = defaultdict(lambda: defaultdict(list))

    def store_metrics(
        self, metrics: Dict[str, float], train_eval: Literal["train", "eval"] = "train"
    ) -> None:
        for key, value in metrics.items():
            self._stored_metrics[train_eval][key].append(value)

    def log(self, logs: Dict[str, float], *args, **kwargs) -> None:
        """
        Log `logs` on the various objects watching training, including stored metrics.

        Args:
            logs (`Dict[str, float]`):
                The values to log.
        """
        # logs either has 'loss' or 'eval_loss'
        train_eval = "train" if "loss" in logs else "eval"
        # Add averaged stored metrics to logs
        for key, metrics in self._stored_metrics[train_eval].items():
            logs[key] = torch.tensor(metrics).mean().item()
        del self._stored_metrics[train_eval]
        return super().log(logs, *args, **kwargs)

    def compute_loss(self, model, inputs, return_outputs=False, **kwargs):
        """
        Compute the combined loss using the existing SFT loss and get_digit_loss.
        To do: make loss only on the model output part.
        """
        labels = inputs.get("labels")
        # Forward pass
        outputs = model(**inputs)
        logits = outputs.get("logits")

        # Compute SFT loss
        sft_loss = outputs.loss

        # Compute custom loss
        loss, stats = get_digit_loss(
            loss_sft=sft_loss,
            logits=logits,
            labels=labels,
            tokenizer=self.tokenizer,
            target_temperature=2.0,
            beta=1.0,
        )

        self.store_metrics(stats, train_eval="train")
        return (loss, outputs) if return_outputs else loss


if __name__ == "__main__":
    from datasets import load_dataset, interleave_datasets
    from transformers import AutoTokenizer, AutoModelForCausalLM
    from trl import (
        ModelConfig,
        SFTConfig,
        ScriptArguments,
        TrlParser,
        get_kbit_device_map,
        get_peft_config,
        get_quantization_config,
    )
    from dataclasses import dataclass, field
    from typing import Optional
    # Add custom argument for preprocessing function
    @dataclass
    class CustomScriptArguments(ScriptArguments):
        preprocess_function: Optional[str] = field(
            default="multi_category", metadata={"help": "Preprocessing function to use"}
        )
    parser = TrlParser((CustomScriptArguments, SFTConfig, ModelConfig))
    script_args, training_args, model_config = parser.parse_args_into_dataclasses()
    ################
    # Model & Tokenizer
    ################
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
        torch_dtype=model_config.torch_dtype,
        use_cache=False if training_args.gradient_checkpointing else True,
        device_map=get_kbit_device_map() if quantization_config is not None else None,
        quantization_config=quantization_config,
    )
    tokenizer = AutoTokenizer.from_pretrained(
        model_config.model_name_or_path, trust_remote_code=model_config.trust_remote_code, use_fast=True, truncation=True, padding='max_length', max_length=training_args.max_seq_length
    ) #
    tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        model_config.model_name_or_path, **model_kwargs #trust_remote_code=model_config.trust_remote_code,
    )


    ##############
    # Load dataset
    ##############
    dataset = load_dataset(script_args.dataset_name, num_proc=8) #split='train[:]',
    EOS_TOKEN = tokenizer.eos_token

    def chat_format(prompt, output):
        if "gemma" in model_config.model_name_or_path:
            if "9b" in model_config.model_name_or_path:
                # return f"<start_of_turn>user {prompt}<end_of_turn>\n<start_of_turn>model {output}<end_of_turn>"
                return tokenizer.apply_chat_template([{"role": "user", "content": prompt}, {"role": "assistant", "content": output}], tokenize=False, add_generation_prompt=False)
            else:
                return f"<|im_start|>user\n{prompt}\n[Score]:<|im_end|><|im_start|>model\n{output}<|im_end|>"
        elif "llama" in model_config.model_name_or_path:
            return tokenizer.apply_chat_template([{"role": "user", "content": prompt}, {"role": "model", "content": output}], tokenize=False, add_generation_prompt=False)
        else:
            return alpaca_prompt.format(prompt, output) + EOS_TOKEN

    def preprocess_function_helpfulness(example):
        prompt_to_helpfulness = helpfulness_prompt.format(example['prompt'], example['response'])

        output_helfulness_score = helpfulness_output_prompt.format(example['helpfulness'])
        #conver to chatml format
        example['text'] = chat_format(prompt_to_helpfulness, output_helfulness_score)
        return example

    def preprocess_function_correctness(example):
        prompt_to_correctness = correctness_prompt.format(example['prompt'], example['response'])

        output_correctness_score = correctness_output_prompt.format(example['correctness'])
        example['text'] = chat_format(prompt_to_correctness, output_correctness_score)
        return example

    def preprocess_function_coherence(example):
        prompt_to_coherence = coherence_prompt.format(example['prompt'], example['response'])

        output_coherence_score = coherence_output_prompt.format(example['coherence'])
        example['text'] = chat_format(prompt_to_coherence, output_coherence_score)
        return example

    def preprocess_function_complexity(example):
        prompt_to_complexity = complexity_prompt.format(example['prompt'], example['response'])

        output_complexity_score = complexity_output_prompt.format(example['complexity'])
        example['text'] = chat_format(prompt_to_complexity, output_complexity_score)
        return example

    def preprocess_function_verbosity(example):
        prompt_to_verbosity = verbosity_prompt.format(example['prompt'], example['response'])

        output_verbosity_score = verbosity_output_prompt.format(example['verbosity'])
        example['text'] = chat_format(prompt_to_verbosity, output_verbosity_score)
        return example

    def preprocess_function_multi_category(example):
        prompt_to_multi_category = overall_prompt.format(example['prompt'], example['response'])

        output_multi_category_score = overall_output_prompt.format(example['helpfulness'], example['correctness'], example['coherence'], example['complexity'], example['verbosity'])
        example['text'] = chat_format(prompt_to_multi_category, output_multi_category_score)
        return example

    def preprocess_function_reasoning(example):
        prompt_to_reasoning = reasoning_prompt.format(example['prompt'], example['response'])
        overall_score = int(example['helpfulness'] + example['correctness'] + example['coherence'] + example['complexity'] + example['verbosity'])
        output_reasoning_score = reasoning_output_prompt.format(overall_score, example['helpfulness'], example['correctness'], example['coherence'], example['complexity'], example['verbosity'])
        example['text'] = chat_format(prompt_to_reasoning, output_reasoning_score)
        return example

    if script_args.preprocess_function == "helpfulness":
        dataset = dataset.map(
            preprocess_function_helpfulness,
            num_proc=8,
            load_from_cache_file=True,
        )
        new_datset_train = dataset['train']
        new_datset_validation = dataset['validation']
    elif script_args.preprocess_function == "multi_category":
        dataset = dataset.map(
            preprocess_function_multi_category,
            num_proc=8,
            load_from_cache_file=True,
        )
        new_datset_train = dataset['train']
        new_datset_validation = dataset['validation']
        print(new_datset_train[0])
    elif script_args.preprocess_function == "reasoning":
        dataset = dataset.map(
            preprocess_function_reasoning,
            num_proc=8,
            load_from_cache_file=True,
        )
        new_datset_train = dataset['train']
        new_datset_validation = dataset['validation']
    elif script_args.preprocess_function == "multi_task":
        dataset_copy = copy.deepcopy(dataset)
        dataset_helpfulness = dataset_copy.map(
            preprocess_function_helpfulness,
            num_proc=8,
            load_from_cache_file=True,
        )
        dataset_copy = copy.deepcopy(dataset)
        dataset_correctness = dataset_copy.map(
            preprocess_function_correctness,
            num_proc=8,
            load_from_cache_file=True,
        )
        dataset_copy = copy.deepcopy(dataset)
        dataset_coherence = dataset_copy.map(
            preprocess_function_coherence,
            num_proc=8,
            load_from_cache_file=True,
        )
        dataset_copy = copy.deepcopy(dataset)
        dataset_complexity = dataset_copy.map(
            preprocess_function_complexity,
            num_proc=8,
            load_from_cache_file=True,
        )
        dataset_copy = copy.deepcopy(dataset)
        dataset_verbosity = dataset_copy.map(
            preprocess_function_verbosity,
            num_proc=8,
            load_from_cache_file=True,
        )
        dataset_copy = copy.deepcopy(dataset)
        dataset_multi_category = dataset_copy.map(
            preprocess_function_multi_category,
            num_proc=8,
            load_from_cache_file=True,
        )
        new_datset_train = interleave_datasets([dataset_helpfulness['train'], dataset_correctness['train'], dataset_coherence['train'], dataset_complexity['train'], dataset_verbosity['train'], dataset_multi_category['train']],
                                        seed=42, probabilities = [0.1, 0.1, 0.1, 0.1, 0.1, 0.5])
        new_datset_validation = interleave_datasets([dataset_helpfulness['validation'], dataset_correctness['validation'], dataset_coherence['validation'], dataset_complexity['validation'], dataset_verbosity['validation'], dataset_multi_category['validation']],
                                        seed=42, probabilities = [0.1, 0.1, 0.1, 0.1, 0.1, 0.5])
    #remove unused columns except text
    # dataset = dataset.remove_columns([col for col in dataset.column_names if col != 'text'])
    # print(dataset.column_names)

    # new_datset = load_dataset(script_args.dataset_name, num_proc=8)

    ##########
    # Training
    ##########
    trainer = Dist2LossTrainer(
        model= model, #model_config.model_name_or_path,
        tokenizer=tokenizer,
        args=training_args,
        train_dataset=new_datset_train,
        eval_dataset=new_datset_validation,
        peft_config=get_peft_config(model_config),
    )
    trainer.train()

    ############################
    # Save model and push to Hub
    ############################
    trainer.save_model(training_args.output_dir)
    metrics = trainer.evaluate()
    trainer.log_metrics("eval", metrics)
    trainer.save_metrics("eval", metrics)