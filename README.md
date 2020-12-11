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

## Experiment Results
Below is a summary of our findings. For full discussion please see our paper:
[Models, Pixels, and Rewards: Evaluating Design Trade-offs in Visual Model-Based Reinforcement Learning](https://arxiv.org/abs/2012.04603) 

Is predicting future rewards sufficient for achieving success in visual 
model-based reinforcement learning? We experimentally demonstrate that this 
is usually **not** the case in the online settings and the key is to 
predict future images too.

![](https://user-images.githubusercontent.com/4112440/101852808-4ef49000-3b13-11eb-9266-8ea3ed291bd9.gif)

Amazingly, this also means there is a weak correlation between reward 
prediction accuracy and performance of the agent. However, we show that there
 is a much stronger correlation between image reconstruction error and the 
 performance of the agent.
 
![](https://user-images.githubusercontent.com/4112440/101852932-9713b280-3b13-11eb-8003-d0080a482872.png)
 
 We show how this phenomenon is directly related to exploration: models that
  fit the data better usually perform better in an *offline* setup. 
  Surprisingly, these are often not the same models that perform the best
   when learning and exploring from scratch!

![](https://user-images.githubusercontent.com/4112440/101853015-c32f3380-3b13-11eb-9823-47befb7745ba.jpeg)


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
