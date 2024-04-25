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