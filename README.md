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
- name: OpenBioLLM-70B
  results: []
license: apache-2.0
language:
- en
datasets:
- berkeley-nest/Nectar
widget:
- example_title: OpenBioLLM-70B 
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



![image/png](https://cdn-uploads.huggingface.co/production/uploads/5f3fe13d79c1ba4c353d0c19/fJIOPJnY6Ff6fUiSIuMEt.png)




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

Introducing OpenBioLLM-70B: A State-of-the-Art Open Source Biomedical Large Language Model


OpenBioLLM-70B is an advanced open source language model designed specifically for the biomedical domain. Developed by Saama AI Labs, this model leverages cutting-edge techniques to achieve state-of-the-art performance on a wide range of biomedical tasks.

🏥 **Biomedical Specialization**: OpenBioLLM-70B is tailored for the unique language and knowledge requirements of the medical and life sciences fields. It was fine-tuned on a vast corpus of high-quality biomedical data, enabling it to understand and generate text with domain-specific accuracy and fluency.

🎓 **Superior Performance**: With 70 billion parameters, OpenBioLLM-70B outperforms other open source biomedical language models of similar scale. It has also demonstrated better results compared to larger proprietary & open-source models like GPT-4,  Gemini, Meditron-70B, Med-PaLM-1 & Med-PaLM-2 on biomedical benchmarks.

🧠 **Advanced Training Techniques**: OpenBioLLM-70B builds upon the powerful foundations of the **Meta-Llama-3-70B-Instruct** and [Meta-Llama-3-70B-Instruct](meta-llama/Meta-Llama-3-70B-Instruct) models. It incorporates the same dataset and fine-tuning recipe as the Starling model, along with a custom diverse medical instruction dataset and a novel merge method. Key components of the training pipeline include:


![image/png](https://cdn-uploads.huggingface.co/production/uploads/5f3fe13d79c1ba4c353d0c19/oPchsJsEpQoGcGXVbh7YS.png)



