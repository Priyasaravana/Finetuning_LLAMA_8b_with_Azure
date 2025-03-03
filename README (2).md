# Fine-Tuning LLama 3.1 8B Instruct Model for Question Answering

## Overview
This repository contains the implementation details for fine-tuning the **LLama 3.1 8B Instruct Model** to perform question-answering tasks. The fine-tuning process was carried out using **LoRA (Low-Rank Adaptation)** and the **PEFT (Parameter-Efficient Fine-Tuning)** library on the **SQuAD (Stanford Question Answering Dataset)**.

## Model Details
- **Base Model**: LLama 3.1 8B Instruct Model
  - https://huggingface.co/meta-llama/Llama-3.1-8B-Instruct
  - meta-llama/Llama-3.1-8B-Instruct
- **Fine-Tuned Task**: Question Answering

### Key Features
- **Parameter Efficient Fine-Tuning**: By using LoRA and PEFT
- **Dataset**: The fine-tuning was conducted on the **SQuAD** dataset, a widely-used benchmark for QA tasks.

## Fine-Tuning Pipeline
1. **Environment Setup**
   - Ensure Python >= 3.10
   - Frameworks Used:
     - Transformers (Hugging Face)
     - PEFT and LoRA libraries

2. **Data Preparation**
   - Dataset: [SQuAD Dataset](https://rajpurkar.github.io/SQuAD-explorer/)
   - Preprocessing involved tokenization and converting data to a suitable format for training.

3. **Fine-Tuning Configuration**
   - Adapter Method: LoRA
   - Batch Size: 2
   - Epochs: 3
   - Optimizer: AdamW
   - Learning Rate: 1e-4

## Results
- Evaluation Metric: F1 Score
- Performance: Achieved competitive results on the validation split taken as test split of SQuAD.
  - **F1 Score**: 86%

## Loading Model 
   ```python
  from transformers import AutoTokenizer, AutoModelForCausalLM
  
  tokenizer = AutoTokenizer.from_pretrained("Preethi-1995/Llama-3-8B-Instruct-SQUAD")
  model = AutoModelForCausalLM.from_pretrained("Preethi-1995/Llama-3-8B-Instruct-SQUAD")
  
  pipe = pipeline(
      task="text-generation",
      model=model,
      tokenizer=tokenizer,
      max_new_tokens=128,
      return_full_text=False,
  )
  
  row = dataset["test"][0]
  prompt = create_test_prompt(row)
  print(prompt)
  
  %%time
  outputs = pipe(prompt)
  response = f"""
  answer:     {row["answer"]}
  prediction: {outputs[0]["generated_text"]}
  """
  print(response)
```
## License
This project is licensed under the MIT License - see the LICENSE file for details.

## Acknowledgements
- Hugging Face for providing the Transformers library.
- Stanford NLP Group for the SQuAD dataset.


