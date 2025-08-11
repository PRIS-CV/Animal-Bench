# 👀 About Animal-Bench
Code release for "Animal-Bench: Benchmarking Multimodal Video Models for Animal-centric Video Understanding"

With the emergence of large pre-trained multimodal video models, multiple benchmarks have been proposed to evaluate model capabilities. However, most of the benchmarks are human-centric, with evaluation data and tasks centered around human applications. Animals are an integral part of the natural world, and animalcentric video understanding is crucial for animal welfare and conservation efforts. Yet, existing benchmarks overlook evaluations focused on animals, limiting the application of the models. To address this limitation, our work established an animal-centric benchmark, namely Animal-Bench, to allow for a comprehensive evaluation of model capabilities in real-world contexts, overcoming agent-bias in previous benchmarks. Animal-Bench includes 13 tasks encompassing both common tasks shared with humans and special tasks relevant to animal conservation, spanning 7 major animal categories and 819 species, comprising a total of 41,839 data entries. To generate this benchmark, we defined a task system centered on animals and proposed an automated pipeline for animal-centric data processing. To further validate the robustness of models against real-world challenges, we utilized a video editing approach to simulate realistic scenarios like weather changes and shooting parameters due to animal movements. We evaluated 8 current multimodal video models on our benchmark and found considerable room for improvement. We hope our work provides insights for the community and opens up new avenues for research in multimodal video models.

<img width="811" height="378" alt="image" src="https://github.com/user-attachments/assets/4fad7eeb-5fd5-4457-9d30-41c3395a868d" />

<img width="800" height="594" alt="image" src="https://github.com/user-attachments/assets/d498b88f-7281-4f0b-bed4-77979d634cb8" />

## Evaluation results

<img width="812" height="469" alt="image" src="https://github.com/user-attachments/assets/2057de50-19d2-43bf-9b4a-ff6dfd11d394" />

## Acknowledgement
Thanks to the open source of the following datasets:
[MammalNet](https://mammal-net.github.io/),[Animal Kingdom](https://sutdcv.github.io/Animal-Kingdom/),[LoTE-Animal](https://lote-animal.github.io/),[MSRVTT-QA](https://github.com/xudejing/video-question-answering?tab=readme-ov-file),[TGIF-QA](https://github.com/YunseokJANG/tgif-qa),[NExT-QA](https://github.com/doc-doc/NExT-QA).
Thanks to the open source of the following models:
[mPLUG-Owl](https://github.com/X-PLUG/mPLUG-Owl),[VideoChat](https://github.com/OpenGVLab/Ask-Anything),[Video-ChatGPT](https://github.com/mbzuai-oryx/Video-ChatGPT),[Video-LLaMA](https://github.com/DAMO-NLP-SG/Video-LLaMA),[Valley](https://github.com/RupertLuo/Valley),[Chat-UniVi](https://github.com/PKU-YuanGroup/Chat-UniVi),[Video-LLaVA](https://github.com/PKU-YuanGroup/Video-LLaVA),[VideoChat2](https://github.com/OpenGVLab/Ask-Anything).
