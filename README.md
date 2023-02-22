# MU-learning examples
This repository contains example scripts using the python module `mu_learning` (`mediated_uncoupled_learning`)
which implements _MU-learning_ methods proposed in [1, 2].

[1] Ikko Yamane, Junya Honda, Florian Yger, and Masashi Sugiyama.
Mediated uncoupled learning: Learning functions without direct input-output correspondences.
In Proceedings of the 38th International Conference on Machine Learning (ICML 2021), 2021. [[arXiv version](https://arxiv.org/abs/2107.08135)]

[2] Ikko Yamane, Yann Chevaleyre, Takashi Ishida, and Florian Yger.
Mediated Uncoupled Learning and Validation with Bregman Divergences: Loss Family with Maximal Generality.
In Proceedings of the 26th International Conference on Artificial Intelligence and Statistics (AISTATS 2023), 2023.


# How to clone this repo
Clone this repo together with the submodules by:
```shell
$ git clone --recurse-submodules git@github.com:i-yamane/mu_learning_examples.git
```

# Execution
See the example shell scripts in the `sh_scripts` directory.

# Browse results
Execute
```shell
$ mlflow ui --backend-store-uri file:<directory_storing_results>
```

