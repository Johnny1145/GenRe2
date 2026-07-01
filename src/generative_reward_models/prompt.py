overall_prompt = """Please act as an impartial judge and evaluate the quality of the response provided by AI assistant to the user question displayed below. Your evaluation should consider five factors helpfulness, correctness, coherence, complexity, verbosity. Here's brief explanation of each factor:
- Helpfulness: Overall helpfulness of the response to the prompt.
- Correctness: Inclusion of all pertinent facts without errors.
- Coherence: Consistency and clarity of expression.
- Complexity: Intellectual depth required to write response (i.e. whether the response can be written by anyone with basic language competency or requires deep domain expertise).
- Verbosity: Amount of detail included in the response, relative to what is asked for in the prompt.
Do not allow the length of the responses to influence your evaluation. Be as objective as possible.
Please provide a set of 5 score over model response. Only provide the score as a number between 0 and 4.

[User Question]
{}

[Start of Model Response]
{}
[End of Model Response]"""

overall_output_prompt = """[Overall Evaluation Score]
helpfulness: {}
correctness: {}
coherence: {}
complexity: {}
verbosity: {}"""

reasoning_prompt = """Please act as an impartial judge and evaluate the quality of the response provided by AI assistant to the user question displayed below. Your evaluation should consider five factors helpfulness, correctness, coherence, complexity, verbosity. Here's brief explanation of each factor:
- Helpfulness: Overall helpfulness of the response to the prompt.
- Correctness: Inclusion of all pertinent facts without errors.
- Coherence: Consistency and clarity of expression.
- Complexity: Intellectual depth required to write response (i.e. whether the response can be written by anyone with basic language competency or requires deep domain expertise).
- Verbosity: Amount of detail included in the response, relative to what is asked for in the prompt.
Do not allow the length of the responses to influence your evaluation. Be as objective as possible.
Please first provide an overall score over model response. You must provide overall score as a number between 0 and 20.
Then provide a set of 5 score over model response. Only provide the score as a number between 0 and 4.

[User Question]
{}

[Start of Model Response]
{}
[End of Model Response]"""

reasoning_output_prompt = """[Overall Evaluation Score]
Overall Score: {}

[Sub-Category Score]
helpfulness: {}
correctness: {}
coherence: {}
complexity: {}
verbosity: {}"""


helpfulness_prompt = """Please act as an impartial judge and evaluate the quality of the response provided by AI assistant to the user question displayed below. Your evaluation should consider how helpful the response is to the prompt. In detail, your evaluation should consider the following:
- Helpfulness: Overall helpfulness of the response to the prompt.
Do not allow the length of the responses to influence your evaluation. Be as objective as possible.
Please provide a score over model response. Only provide the score as a number between 0 and 4.

[User Question]
{}

[Start of Model Response]
{}
[End of Model Response]"""

helpfulness_output_prompt = """[Helpfulness Evaluation Score]
helpfulness: {}"""

overall_prompt_tie_breaker = """<|begin_of_text|><|start_header_id|>system<|end_header_id|>\n\nCutting Knowledge Date: December 2023\nToday Date: 26 Jul 2024\n\n<|eot_id|><|start_header_id|>user<|end_header_id|>\n\nPlease act as an impartial judge and evaluate the quality of the response provided by AI assistant to the user question displayed below. Your evaluation should consider five factors helpfulness, correctness, coherence, complexity, verbosity. Here's brief explanation of each factor:
- Helpfulness: Overall helpfulness of the response to the prompt.
- Correctness: Inclusion of all pertinent facts without errors.
- Coherence: Consistency and clarity of expression.
- Complexity: Intellectual depth required to write response (i.e. whether the response can be written by anyone with basic language competency or requires deep domain expertise).
- Verbosity: Amount of detail included in the response, relative to what is asked for in the prompt.
Do not allow the length of the responses to influence your evaluation. Be as objective as possible.
Please provide a set of 5 score over model response. Only provide the score as a number between 0 and 4.

[User Question]
{}

[Start of Model Response]
{}
[End of Model Response]<|eot_id|><|start_header_id|>assistant<|end_header_id|>\n\n[Overall Evaluation Score]\nhelpfulness: """

reasoning_prompt_tie_breaker = """<|begin_of_text|><|start_header_id|>system<|end_header_id|>\n\nCutting Knowledge Date: December 2023\nToday Date: 26 Jul 2024\n\n<|eot_id|><|start_header_id|>user<|end_header_id|>\n\nPlease act as an impartial judge and evaluate the quality of the response provided by AI assistant to the user question displayed below. Your evaluation should consider five factors helpfulness, correctness, coherence, complexity, verbosity. Here's brief explanation of each factor:
- Helpfulness: Overall helpfulness of the response to the prompt.
- Correctness: Inclusion of all pertinent facts without errors.
- Coherence: Consistency and clarity of expression.
- Complexity: Intellectual depth required to write response (i.e. whether the response can be written by anyone with basic language competency or requires deep domain expertise).
- Verbosity: Amount of detail included in the response, relative to what is asked for in the prompt.
Do not allow the length of the responses to influence your evaluation. Be as objective as possible.
Please provide a set of 5 score over model response. Only provide the score as a number between 0 and 4.

[User Question]
{}

[Start of Model Response]
{}
[End of Model Response]<|eot_id|><|start_header_id|>assistant<|end_header_id|>\n\n[Overall Evaluation Score]\nOverall Score: """
