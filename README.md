# RolyPlaying
```bib
The resource code is from [In-Context Impersonation on GitHub](https://github.com/ExplainableML/in-context-impersonation)

@article{Salewski2023InContextIR,
  title   = {In-Context Impersonation Reveals Large Language Models' Strengths and Biases},
  author  = {Leonard Salewski and Stephan Alaniz and Isabel Rio-Torto and Eric Schulz and Zeynep Akata},
  journal = {ArXiv},
  year    = {2023},
  volume  = {abs/2305.14930},
}
```
To run the code:
1. Apply for llama token in huggingface page.
2. Use "huggingface-cli login" to login huggingface and input your token.
3. Creat conde environment: conda env create -f environment.yaml -n RolePlaying // conda activate RolePlaying
4. Create a folder to store the model config like "RolePlaying/shared/llama3/1B" 
5. run "python3 download.py" to load the corresponding model, it will be stored in "RolePlaying/shared"
6. run the code by python3 src/eval.py model=text_otf data=mmlu data.dataset_partial.task=abstract_algebra
