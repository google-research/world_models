# World Models Library

World Models is a platform-agnostic library to facilitate visual based 
agents for planning. This notebook 
([run it in colab](https://colab.research.google.com/github/google-research/world_models/blob/master/intro.ipynb))
shows how to use World Models library and its different
components.

To run locally, use the following command:

```$xslt
python3 -m world_models.bin.train_eval  \
    --config_path=/path/to/config  \
    --output_dir=/path/to/output_dir  \
    --logtostderr
```

## How to Cite
If you use this work, please cite the following paper where it was first introduced:
```   
   @article{2020worldmodels,
     title   = {Models, Pixels, and Rewards: Evaluating Design Trade-offs in Visual Model-Based Reinforcement Learning},
     author  = {Mohammad Babaeizadeh and Mohammad Taghi Saffar and Danijar Hafner and Harini Kannan and Chelsea Finn and Sergey Levine and Dumitru Erhan},
     year    = {2020},
     url     = {https://arxiv.org/abs/2012.04603}
   }
```

You can reach us at wm-core@google.com 
## Dependencies
* absl
* gin-config
* TensorFlow==1.15
* TensorFlow probability==0.7
* gym
* dm_control
* MuJoCo

Disclaimer: This is not an official Google product.
