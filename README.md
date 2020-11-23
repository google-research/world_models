# World Models Library

World Models is a platform-agnostic library to facilitate visual based 
agents for planning. This notebook 
([run it in colab](https://colab.research.google.com/github/google_research/world_models/blob/master/intro.ipynb))
shows how to use World Models library and its different
components.

To run locally, use the following command:

```$xslt
python3 -m world_models.bin.train_eval  \
    --config_path=/path/to/config  \
    --output_dir=/path/to/output_dir  \
    --logtostderr
```

## Dependencies
* absl
* gin-config
* TensorFlow==1.15
* TensorFlow probability==0.7
* gym
* dm_control
* MuJoCo

