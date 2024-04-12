# ImageCLEF Medical: Concept Prediction and Captioning

Scripts, figures, and working notes for the participation in [ImageCLEFmed Caption task](https://www.imageclef.org/2023/medical/caption), part of the [ImageCLEF labs](https://www.imageclef.org/2023) at the [14th CLEF Conference, 2023](https://clef2023.clef-initiative.eu/index.php).

**Implementation Stack**: Python, Keras, TensorFlow, Scikit-learn.

## Quick Links

- [Link to the research paper](https://ceur-ws.org/Vol-3497/paper-136.pdf) describing the approach, rationale, and results.
- [Dataset for 2023](https://drive.google.com/drive/folders/14GmtlRUQ1LDnO9PkpSjIA6eJW8lLVuT7?usp=share_link): training, validation, and testing sets.
- [Link to contest resources](https://www.imageclef.org/2023/medical/caption).
- [Link to shared Google Drive [**private access**]](https://drive.google.com/drive/folders/1fd0SRO2IColNpPwsecNiNSeI7E3spad6?usp=sharing). 


## Cite Us

[Link to the Research Paper](https://ceur-ws.org/Vol-3497/paper-136.pdf).

If you find our work useful in your research, don't forget to cite us!

```
@article{palaniappan2023concept,
  url = {https://ceur-ws.org/Vol-3497/paper-136.pdf},
  title={Concept Detection and Caption Prediction from Medical Images using Gradient Boosted Ensembles and Deep Learning},
  author={Palaniappan, Mirunalini and Bharathi, Haricharan and Chodisetty, Eeswara Anvesh and Bhaskar, Anirudh and Desingu, Karthik},
  year={2023},
  keywords={concept detection, caption prediction, natural language processing, computer vision ensemble, feature extraction, deep learning, automated image captioning},
  journal={Conference and Labs of the Evaluation Forum},
  publisher={Conference and Labs of the Evaluation Forum},
  ISSN={1613-0073},  
  copyright = {Creative Commons Attribution 4.0 International}
}
```

## Objectives

To develop automatic methods that can approximate a mapping from visual information to condensed textual descriptions, towaed interpreting and summarizing the insights gained from medical images such as radiology outputs and histology slides. To this end, the contest breaks this objective down to a sequence of two tasks.

### Task A: Concept Detection

Automatic image captioning and scene understanding to identify the presence and location of relevant concepts in the given medical image.
Concepts are derived from the [UMLS medical metathesaurus](https://www.nlm.nih.gov/research/umls/index.html).

### Task B: Caption Predcition

On the basis of the concept vocabulary detected in the first subtask as well as the visual information of their interaction in the image, this task is aimed at composing coherent captions for the entirety of an image. 

*[Link to contest page](https://www.imageclef.org/2023/medical/caption), for a more detailed description.*

## Reproducing the Implementation

- The *conda* environment with all dependencies can be set up using [environment.yml](./environment.yml).
- All relevant code, scripts, and python notebooks are contained in [scripts](./scripts).
- Resources for tasks A and B are prefixed with **A** and **B**, respectively.
- Simply run the scripts for each task serially; the order is indicated as filename prefixes.
