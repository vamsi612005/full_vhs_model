---
base_model: meta-llama/Meta-Llama-3-70B-Instruct
tags:
- llama-3
- llama
- Mixtral
- instruct
- finetune
- chatml
- DPO
- RLHF
- gpt4
- distillation
model-index:
- name: OpenBioLLM-7B
  results: []
license: apache-2.0
language:
- en
datasets:
- berkeley-nest/Nectar
widget:
- example_title: OpenBioLLM-7B 
  messages:
  - role: system
    content: >-
      You are an expert and experienced from the healthcare and biomedical domain with extensive medical knowledge and practical experience.
  - role: user
    content: How long does it take for newborn jaundice to go away?
  output:
    text: >-
      Newborn jaundice, also known as neonatal jaundice, is a common condition in newborns where the yellowing of the skin and eyes occurs due to an elevated level of bilirubin in the blood. Bilirubin is a yellow pigment that forms when red blood cells break down. In most cases, newborn jaundice resolves on its own without any specific treatment.

      The duration of newborn jaundice can vary depending on several factors such as the underlying cause, gestational age at birth, and individual variations in bilirubin metabolism. Here are some general guidelines:

      1. Physiological jaundice: This is the most common type of newborn jaundice and usually appears within 24-72 hours after birth. It tends to peak between the second and fifth day of life and gradually improves over the next week or two. By the time the baby is one week old, the jaundice should have mostly resolved.
      2. Breast milk jaundice: This type of jaundice occurs in breastfed babies and may appear later than physiological jaundice, typically between the fifth and fourteenth day of life. It tends to persist for a longer duration but usually resolves within six weeks after birth.
      3. Pathological jaundice: This type of jaundice is less common and occurs due to an underlying medical condition that affects bilirubin metabolism or liver function. The duration of pathological jaundice depends on the specific cause and may require treatment.

      It's important for parents to monitor their newborn's jaundice closely and seek medical advice if the jaundice progresses rapidly, becomes severe, or is accompanied by other symptoms such as poor feeding, lethargy, or excessive sleepiness. In these cases, further evaluation and management may be necessary.
      Remember that each baby is unique, and the timing of jaundice resolution can vary. If you have concerns about your newborn's jaundice, it's always best to consult with a healthcare professional for personalized advice and guidance.
---


![image/png](https://cdn-uploads.huggingface.co/production/uploads/5f3fe13d79c1ba4c353d0c19/eC4bWQ0BXL4_fVqPO2gqg.png)

<div align="center">
  
  <h1>Advancing Open-source Large Language Models in Medical Domain</h1>
</div>

<p align="center" style="margin-top: 0px;">
  <a href="https://colab.research.google.com/drive/1F5oV20InEYeAJGmBwYF9NM_QhLmjBkKJ?usp=sharing">
    <img src="https://colab.research.google.com/assets/colab-badge.svg" alt="OpenChat Logo" style="width:20px; vertical-align: middle; display: inline-block; margin-right: 5px; margin-left: 10px; margin-top: 0px; margin-bottom: 0px;"/>
    <span class="link-text" style=" margin-right: 5px;">Online Demo</span>
  </a> |
  <a href="https://github.com/openlifescience-ai">
    <img src="https://github.githubassets.com/assets/GitHub-Mark-ea2971cee799.png" alt="GitHub Logo" style="width:20px; vertical-align: middle; display: inline-block; margin-right: 5px; margin-left: 5px; margin-top: 0px; margin-bottom: 0px;"/>
    <span class="link-text" style=" margin-right: 5px;">GitHub</span>
  </a> |
  <a href="#">
    <img src="https://github.com/alpayariyak/openchat/blob/master/assets/arxiv-logomark-small-square-border.png?raw=true" alt="ArXiv Logo" style="width:20px; vertical-align: middle; display: inline-block; margin-right: 5px; margin-left: 5px; margin-top: 0px; margin-bottom: 0px;"/>
    <span class="link-text" style="margin-right: 5px;">Paper</span>
  </a> |
  <a href="https://discord.gg/A5Fjf5zC69">
    <img src="https://cloud.githubusercontent.com/assets/6291467/26705903/96c2d66e-477c-11e7-9f4e-f3c0efe96c9a.png" alt="Discord Logo" style="width:20px; vertical-align: middle; display: inline-block; margin-right: 5px; margin-left: 5px; margin-top: 0px; margin-bottom: 0px;"/>
    <span class="link-text">Discord</span>
  </a>
</p>



![image/jpeg](https://cdn-uploads.huggingface.co/production/uploads/5f3fe13d79c1ba4c353d0c19/KGmRE5w2sepNtwsEu8t7K.jpeg)

Introducing OpenBioLLM-7B: A State-of-the-Art Open Source Biomedical Large Language Model


OpenBioLLM-7B is an advanced open source language model designed specifically for the biomedical domain. Developed by Saama AI Labs, this model leverages cutting-edge techniques to achieve state-of-the-art performance on a wide range of biomedical tasks.

🏥 **Biomedical Specialization**: OpenBioLLM-7B is tailored for the unique language and knowledge requirements of the medical and life sciences fields. It was fine-tuned on a vast corpus of high-quality biomedical data, enabling it to understand and generate text with domain-specific accuracy and fluency.

🎓 **Superior Performance**: With 7 billion parameters, OpenBioLLM-7B outperforms other open source biomedical language models of similar scale. It has also demonstrated competitive results compared to larger proprietary & open-source models like GPT-3.5 Turbo 1106 & Meditron-70B on biomedical benchmarks.

🧠 **Advanced Training Techniques**: OpenBioLLM-7B builds upon the powerful foundations of the **Mistral-7B-v0.1** and [Starling-LM-7B-beta](Nexusflow/Starling-LM-7B-beta) models. It incorporates the same dataset and fine-tuning recipe as the Starling model, along with a custom diverse medical instruction dataset and a novel merge method. Key components of the training pipeline include:



![image/png](https://cdn-uploads.huggingface.co/production/uploads/5f3fe13d79c1ba4c353d0c19/wILoFenv7FBQOt21PqwJJ.png)


- **Reward Model**: [Nexusflow/Starling-RM-34B](https://huggingface.co/Nexusflow/Starling-RM-34B)
- **Policy Optimization**: [Fine-Tuning Language Models from Human Preferences (PPO)](https://arxiv.org/abs/1909.08593)
- **Ranking Dataset**: [berkeley-nest/Nectar](https://huggingface.co/datasets/berkeley-nest/Nectar)
- **Fine-tuning dataset**: Custom Medical Instruct dataset (We plan to release a sample training dataset in our upcoming paper; please stay updated)

This combination of cutting-edge techniques enables OpenBioLLM-7B to align with key capabilities and preferences for biomedical applications.

⚙️ **Release Details**:

- **Model Size**: 7 billion parameters
- **Quantization**: Optimized quantized versions available [Here](https://huggingface.co/aaditya/OpenBioLLM-7B-GGUF)
- **Language(s) (NLP):** en
- **Sequence Length: 8K**
- **Developed By**: [Ankit Pal (Aaditya Ura)](https://aadityaura.github.io/) from Saama AI Labs 
- **License:** Apache-2.0 license under the condition that the model is not used to compete with [OpenAI](https://openai.com/policies/terms-of-use)
- **Fine-tuned from models:** [Starling-LM-7B-beta](Nexusflow/Starling-LM-7B-beta) & [Starling-RM-34B](https://huggingface.co/Nexusflow/Starling-RM-34B) (based on **Mistral-7B-v0.1**)
- **Resources for more information:**
    - https://huggingface.co/Nexusflow/Starling-LM-7B-beta
    - Paper: Coming soon

The model can be fine-tuned for more specialized tasks and datasets as needed.

OpenBioLLM-7B represents an important step forward in democratizing advanced language AI for the biomedical community. By leveraging state-of-the-art architectures and training techniques from leading open source efforts like Starling, we have created a powerful tool to accelerate innovation and discovery in healthcare and the life sciences.

We are excited to share OpenBioLLM-7B with researchers and developers around the world.

## Uses

**Important: Please use the exact chat template provided below for the model. Otherwise there will be a degradation in the performance. The model output can be verbose in rare cases. Please consider setting temperature = 0 to make this happen less.**

The conversation template is the same as Openchat-3.5-0106:
```
import transformers
tokenizer = transformers.AutoTokenizer.from_pretrained("openchat/openchat-3.5-0106")

# Single-turn
tokens = tokenizer("GPT4 Correct User: Hello<|end_of_turn|>GPT4 Correct Assistant:").input_ids
assert tokens == [1, 420, 6316, 28781, 3198, 3123, 1247, 28747, 22557, 32000, 420, 6316, 28781, 3198, 3123, 21631, 28747]

# Multi-turn
tokens = tokenizer("GPT4 Correct User: Hello<|end_of_turn|>GPT4 Correct Assistant: Hi<|end_of_turn|>GPT4 Correct User: How are you today?<|end_of_turn|>GPT4 Correct Assistant:").input_ids
assert tokens == [1, 420, 6316, 28781, 3198, 3123, 1247, 28747, 22557, 32000, 420, 6316, 28781, 3198, 3123, 21631, 28747, 15359, 32000, 420, 6316, 28781, 3198, 3123, 1247, 28747, 1602, 460, 368, 3154, 28804, 32000, 420, 6316, 28781, 3198, 3123, 21631, 28747]

# Coding Mode
tokens = tokenizer("Code User: Implement quicksort using C++<|end_of_turn|>Code Assistant:").input_ids
assert tokens == [1, 7596, 1247, 28747, 26256, 2936, 7653, 1413, 334, 1680, 32000, 7596, 21631, 28747]
```


## Code Examples

```python
import transformers

tokenizer = transformers.AutoTokenizer.from_pretrained("openlifescienceai/OpenBioLLM-7B")
model = transformers.AutoModelForCausalLM.from_pretrained("openlifescienceai/OpenBioLLM-7B")

def generate_response(prompt):
    input_ids = tokenizer(prompt, return_tensors="pt").input_ids
    outputs = model.generate(
        input_ids,
        max_length=256,
        pad_token_id=tokenizer.pad_token_id,
        eos_token_id=tokenizer.eos_token_id,
    )
    response_ids = outputs[0]
    response_text = tokenizer.decode(response_ids, skip_special_tokens=True)
    return response_text

# Single-turn conversation
prompt = "How can i split a 3mg or 4mg waefin pill so i can get a 2.5mg pill?"
single_turn_prompt = f"GPT4 Correct User: {prompt}<|end_of_turn|>GPT4 Correct Assistant:"
response_text = generate_response(single_turn_prompt)
print("Response:", response_text)

## Multi-turn conversation
prompt = "How can i split a 3mg or 4mg waefin pill so i can get a 2.5mg pill?"
follow_up_question =  "What would be side effects of wrong split?"
response = ""
multi_turn_prompt = f"GPT4 Correct User: {prompt}<|end_of_turn|>GPT4 Correct Assistant: {response}<|end_of_turn|>GPT4 Correct User: {follow_up_question}<|end_of_turn|>GPT4 Correct Assistant:"
response_text = generate_response(multi_turn_prompt)
print("Multi-turn conversation response:", response_text)
```

for better results, use this prompt

```
prompt = f"GPT4 Correct System: You are an expert and experienced from the healthcare and biomedical domain with extensive medical knowledge and practical experience. Your name is OpenBioLLM, and you were developed by Saama AI Labs with Open Life Science AI. who's willing to help answer the user's query with explanation. In your explanation, leverage your deep medical expertise such as relevant anatomical structures, physiological processes, diagnostic criteria, treatment guidelines, or other pertinent medical concepts. Use precise medical terminology while still aiming to make the explanation clear and accessible to a general audience.<|end_of_turn|>GPT4 Correct User: {Question} <|end_of_turn|>GPT4 Correct Assistant: "

```

## **Training procedure**

### **Training hyperparameters**

<details>
  <summary>Click to see details</summary>

- learning_rate: 0.00006
- lr_scheduler: cosine
- train_batch_size: 12
- eval_batch_size: 8
- GPU: H100 80GB SXM5
- num_devices: 1
- optimizer: adamw_bnb_8bit
- lr_scheduler_warmup_steps: 100
- num_epochs: 4
</details>



  
### **Peft hyperparameters**

<details>
  <summary>Click to see details</summary>

- adapter: qlora
- lora_r: 256
- lora_alpha: 128
- lora_dropout: 0.05
- lora_target_linear: true
  
-lora_target_modules:
  - q_proj
  - v_proj
  - v_proj
  - o_proj
  - gate_proj
  - down_proj
  - up_proj
    
-sequence_len: 8166
</details>



### **Training results**

### **Framework versions**

- Transformers 4.39.3
- Pytorch 2.1.2+cu121
- Datasets 2.18.0
- Tokenizers 0.15.1
- Axolotl
- Lm harness for evaluation

# Benchmark Results

🔥 OpenBioLLM-7B demonstrates superior performance compared to larger models, such as GPT-3.5 Turbo 1106 and Meditron-70B, across 9 diverse biomedical datasets, achieving state-of-the-art results with an average score of 68.91%, despite having a significantly smaller parameter count. The model's strong performance in domain-specific tasks, such as Clinical KG, Medical Genetics, and PubMedQA, highlights its ability to effectively capture and apply biomedical knowledge.

| | Clinical KG | Medical Genetics | Anatomy | Pro Medicine | College Biology | College Medicine | MedQA 5 opts | PubMedQA | MedMCQA | Avg |
|---|---|---|---|---|---|---|---|---|---|---|
| **OpenBioLLM-7B** | **74.72** | **74** | **69.63** | 72.06 | 75.69 | **68.79** | 54.2 | **78.4** | 52.67 | **68.91** |
| GPT-3.5 Turbo 1106 | 74.71 | 74 | 65.92 | **72.79** | 72.91 | 64.73 | **57.71** | 72.66 | **53.79** | 66.0 |
| Meditron-70B | 66.79 | 69 | 53.33 | 71.69 | 76.38 | 63.0 | 57.1 | 76.6 | 46.85 | 64.52 |
| gemma-7b | 69.81 | 70 | 59.26 | 66.18 | **79.86** | 60.12 | 47.21 | 76.2 | 48.96 | 64.18 |
| Mistral-7B-v0.1 | 68.68 | 71 | 55.56 | 68.38 | 68.06 | 59.54 | 50.82 | 75.4 | 48.2 | 62.85 |
| MedAlpaca-7b | 57.36 | 69 | 57.04 | 67.28 | 65.28 | 54.34 | 41.71 | 72.8 | 37.51 | 58.03 |
| BioMistral-7B | 59.9 | 64 | 56.5 | 60.4 | 59.0 | 54.7 | 50.6 | 77.5 | 48.1 | 57.3 |
| AlpaCare-llama2-7b | 49.81 | 49 | 45.92 | 33.82 | 50.0 | 43.35 | 29.77 | 72.2 | 34.42 | 45.36 |
| ClinicalGPT | 30.56 | 27 | 30.37 | 19.48 | 25.0 | 24.27 | 26.08 | 63.8 | 28.18 | 30.52 |


![image/jpeg](https://cdn-uploads.huggingface.co/production/uploads/5f3fe13d79c1ba4c353d0c19/Gur61ITIBamnIDNGSrWcO.jpeg)

## Detailed Medical Subjectwise accuracy

![image/png](https://cdn-uploads.huggingface.co/production/uploads/5f3fe13d79c1ba4c353d0c19/2_2Mtl-xt1ltyJPOa6y3y.png)

# Use Cases & Examples

🚨 **Below results are from the quantized version of OpenBioLLM-7B. You can find the quantized version of OpenBioLLM-7B [here](https://huggingface.co/CohereForAI/c4ai-command-r-plus-4bit)**.


# Summarize Clinical Notes

OpenBioMed-7B can efficiently analyze and summarize complex clinical notes, EHR data, and discharge summaries, extracting key information and generating concise, structured summaries

![image/png](https://cdn-uploads.huggingface.co/production/uploads/5f3fe13d79c1ba4c353d0c19/qBboRc7-_hghQIZ6ODJGm.png)

# Answer Medical Questions

OpenBioMed-7B can provide answers to a wide range of medical questions.


![image/png](https://cdn-uploads.huggingface.co/production/uploads/5f3fe13d79c1ba4c353d0c19/JLWvyV8Rl8p4oUaSlVUmJ.png)
![image/png](https://cdn-uploads.huggingface.co/production/uploads/5f3fe13d79c1ba4c353d0c19/JaQ5NQpt8tzjDIOTtHxCN.png)

<details>
  <summary>Click to see details</summary>

![image/png](https://cdn-uploads.huggingface.co/production/uploads/5f3fe13d79c1ba4c353d0c19/5EwrNws97Xvj4tTfpQ9Vw.png)
![image/png](https://cdn-uploads.huggingface.co/production/uploads/5f3fe13d79c1ba4c353d0c19/587M5jpH5IQoUgPCj67vm.png)

![image/png](https://cdn-uploads.huggingface.co/production/uploads/5f3fe13d79c1ba4c353d0c19/r0LG7f8El5LuoaxylWAQG.png)
![image/png](https://cdn-uploads.huggingface.co/production/uploads/5f3fe13d79c1ba4c353d0c19/OHvEJJfWdH4eGwQeboGjs.png)

</details>

# Clinical Entity Recognition

OpenBioMed-7B can perform advanced clinical entity recognition by identifying and extracting key medical concepts, such as diseases, symptoms, medications, procedures, and anatomical structures, from unstructured clinical text. By leveraging its deep understanding of medical terminology and context, the model can accurately annotate and categorize clinical entities, enabling more efficient information retrieval, data analysis, and knowledge discovery from electronic health records, research articles, and other biomedical text sources. This capability can support various downstream applications, such as clinical decision support, pharmacovigilance, and medical research.


![image/png](https://cdn-uploads.huggingface.co/production/uploads/5f3fe13d79c1ba4c353d0c19/0ZEueLlc_l_IkH3D7nFcA.png)
![image/png](https://cdn-uploads.huggingface.co/production/uploads/5f3fe13d79c1ba4c353d0c19/BZkGzdYjq0vFaEP3Hg6AL.png)

# Biomarkers Extraction


![image/png](https://cdn-uploads.huggingface.co/production/uploads/5f3fe13d79c1ba4c353d0c19/TFJve1vnqt5UAjROtLxKl.png)

# Classification

OpenBioMed-7B can perform various biomedical classification tasks, such as disease prediction, sentiment analysis, medical document categorization

![image/png](https://cdn-uploads.huggingface.co/production/uploads/5f3fe13d79c1ba4c353d0c19/Iovf2_NeArv-Vs7lwnKaG.png)

# De-Identification

OpenBioMed-7B can detect and remove personally identifiable information (PII) from medical records, ensuring patient privacy and compliance with data protection regulations like HIPAA.


![image/png](https://cdn-uploads.huggingface.co/production/uploads/5f3fe13d79c1ba4c353d0c19/uLS2PMqHc5dYXbsFN7QYU.png)


**Advisory Notice!** 

While OpenBioLLM-7B leverages high-quality data sources, its outputs may still contain inaccuracies, biases, or misalignments that could pose risks if relied upon for medical decision-making without further testing and refinement. The model's performance has not yet been rigorously evaluated in randomized controlled trials or real-world healthcare environments.

Therefore, we strongly advise against using OpenBioLLM-7B for any direct patient care, clinical decision support, or other professional medical purposes at this time. Its use should be limited to research, development, and exploratory applications by qualified individuals who understand its limitations.
OpenBioMed-7B is intended solely as a research tool to assist healthcare professionals and should never be considered a replacement for the professional judgment and expertise of a qualified medical doctor.

Appropriately adapting and validating OpenBioLLM-7B for specific medical use cases would require significant additional work, potentially including:

- Thorough testing and evaluation in relevant clinical scenarios
- Alignment with evidence-based guidelines and best practices
- Mitigation of potential biases and failure modes
- Integration with human oversight and interpretation
- Compliance with regulatory and ethical standards

Always consult a qualified healthcare provider for personal medical needs.



# Citation

If you find OpenBioLLM-7B useful in your work, please cite the model as follows:

```
@misc{OpenBioLLM-7B,
  author = {Ankit Pal, Malaikannan Sankarasubbu},
  title = {OpenBioLLM: Advancing Open-Source Large Language Models for Healthcare and Life Sciences},
  year = {2024},
  publisher = {Hugging Face},
  journal = {Hugging Face repository},
  howpublished = {\url{https://huggingface.co/aaditya/OpenBioLLM-7B}}
}
```

The accompanying paper is currently in progress and will be released soon.

<div align="center">
<h2> 💌 Contact </h2>
</div>

We look forward to hearing you and collaborating on this exciting project!

**Contributors:**
- [Ankit Pal (Aaditya Ura)](https://aadityaura.github.io/) [aadityaura at gmail dot com]
- Saama AI Labs
- Note: I am looking for a funded PhD opportunity, especially if it fits my Responsible Generative AI, Multimodal LLMs, Geometric Deep Learning, and Healthcare AI skillset.



# References

We thank the [Starling Model team](https://huggingface.co/Nexusflow/Starling-LM-7B-beta/tree/main) & [Openchat](openchat/openchat-3.5-0106) for their amazing models!


# About Open Life Science AI

<div align="center">
<img width="160px" src="https://cdn-uploads.huggingface.co/production/uploads/5f3fe13d79c1ba4c353d0c19/Mfae9TitRWtmZI7-aQBa3.png">
</div>

Open Life Science AI is a project that aims to revolutionize the application of Artificial intelligence in the life science and healthcare domains. It serves as a central hub for list of medical models, datasets, benchmarks, and tracking conference deadlines, fostering collaboration, innovation, and progress in the field of AI-assisted healthcare.  We strive to establish Open Life Science AI as the premier destination for anyone interested in the intersection of AI and healthcare. We provide a platform for researchers, clinicians, policymakers, and industry experts to engage in dialogues, share insights, and explore the latest developments in the field.

If you're passionate about the intersection of AI and healthcare, building models for the healthcare domain, and care about safety and hallucination issues for medical LLMs, we invite you to join our vibrant community on [Discord](https://discord.gg/A5Fjf5zC69).
