import os
import torch
from dataclasses import dataclass, field
from datasets import load_dataset
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import PeftModel
from trl import ModelConfig, TrlParser
from tqdm import tqdm
import json

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
    model_path: str = field(
        default="./outputs/sft_Mistral_test",
        metadata={"help": "Path to the trained model (LoRA adapter or merged model)"}
    )
    base_model_path: str = field(
        default=None,
        metadata={"help": "Base model path if model_path is a LoRA adapter"}
    )
    output_json: str = field(
        default="generated_feedback_dataset.json",
        metadata={"help": "Output JSON file path"}
    )
    max_new_tokens: int = field(
        default=1024,
        metadata={"help": "Max new tokens for generation"}
    )
    batch_size: int = field(
        default=4,
        metadata={"help": "Batch size for generation"}
    )
    test_mode: bool = field(
        default=False,
        metadata={"help": "If true, only process a small subset"}
    )

if __name__ == "__main__":
    parser = TrlParser((ScriptArguments, ModelConfig))
    script_args, model_config = parser.parse_args_into_dataclasses()

    print(f"Loading tokenizer from {model_config.model_name_or_path}...")
    tokenizer = AutoTokenizer.from_pretrained(
        model_config.model_name_or_path,
        trust_remote_code=model_config.trust_remote_code,
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "left" # For batch inference

    print(f"Loading model from {model_config.model_name_or_path}...")
    torch_dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16

    model = AutoModelForCausalLM.from_pretrained(
        model_config.model_name_or_path,
        torch_dtype=torch_dtype,
        device_map="auto",
        trust_remote_code=model_config.trust_remote_code,
    )

    #  adapter  LoRA

    model.eval()

    # 3.
    print("Loading Feedback-Collection dataset...")
    ds = load_dataset("prometheus-eval/Feedback-Collection")
    dataset = ds['train']

    # if script_args.test_mode:
    #     print("Test mode: processing only 10 samples")
    #     dataset = dataset.select(range(10))

    # 4.
    new_data = []

    def get_prompt(example):
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
        prompt = INPUT_TEMPLATE.format(**input_kwargs)
        #  chat template
        messages = [{"role": "user", "content": prompt}]
        return tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)

    print(f"Starting generation for {len(dataset)} samples...")

    for i in tqdm(range(0, len(dataset), script_args.batch_size)):
        batch_examples = dataset.select(range(i, min(i + script_args.batch_size, len(dataset))))
        prompts = [get_prompt(ex) for ex in batch_examples]

        inputs = tokenizer(prompts, return_tensors="pt", padding=True).to(model.device)

        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_new_tokens=script_args.max_new_tokens,
                do_sample=False, #
                pad_token_id=tokenizer.pad_token_id,
            )

        #  prompt
        input_len = inputs.input_ids.shape[1]
        generated_ids = outputs[:, input_len:]
        responses = tokenizer.batch_decode(generated_ids, skip_special_tokens=True)

        for ex, resp in zip(batch_examples, responses):
            # 1.  ( [RESULT] )
            clean_feedback = resp.split("[RESULT]")[0].strip()

            # 2.  (Ground Truth)
            gt_score = ex.get('orig_score', ex.get('score', 5))

            # 3.  "So the overall score is"  GT
            phrase = "So the overall score is"
            if phrase in clean_feedback:
                main_body = clean_feedback.split(phrase)[0].strip()
                aligned_feedback = f"{main_body}\n{phrase} {gt_score}"
            else:
                aligned_feedback = f"{clean_feedback}\n{phrase} {gt_score}"

            # 4.  sft_Mistral.py  format_for_sft
            item = {
                #
                "instruction": ex.get('orig_instruction', ex.get('instruction', '')),
                "output": ex.get('orig_response', ex.get('output', ex.get('response', ''))),
                "reference_answer": ex.get('orig_reference_answer', ex.get('reference_answer', "N/A")),
                "rubric": ex.get('orig_criteria', ex.get('rubric', "")),

                #  ()
                "orig_score1_description": ex.get('orig_score1_description', "N/A"),
                "orig_score2_description": ex.get('orig_score2_description', "N/A"),
                "orig_score3_description": ex.get('orig_score3_description', "N/A"),
                "orig_score4_description": ex.get('orig_score4_description', "N/A"),
                "orig_score5_description": ex.get('orig_score5_description', "N/A"),

                #  feedback
                "feedback": aligned_feedback,
                "score": gt_score,

                #
                "orig_feedback": aligned_feedback,
                "raw_feedback": ex.get('orig_feedback', ex.get('feedback', "")),
                "model_raw_response": resp,
            }
            new_data.append(item)

    # 5.
    print(f"Saving new dataset to {script_args.output_json}...")
    with open(script_args.output_json, "w", encoding="utf-8") as f:
        json.dump(new_data, f, ensure_ascii=False, indent=4)

    print("Done!")
