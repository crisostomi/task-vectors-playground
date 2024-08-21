# This scripts runs an entirement, it goes from order 1 all the way to order n.

import yaml
import subprocess

epoch_divisor = 2
desired_orders = 8

yaml_file = "conf/nn/data/default.yaml"
ft_conf_file = "conf/finetune.yaml"
tv_conf_file = "conf/task_vectors.yaml"

for order in range(1, desired_orders+1):

    # adjust hyperparameters in finetune.yaml
    with open(ft_conf_file, "r") as file:
            config = yaml.safe_load(file)
            config['epoch_divisor'] = epoch_divisor
            config['order'] = order
            print(config)
    with open(ft_conf_file, "w") as file:
        yaml.dump(config, file)

    # adjust hyperparameters in task_vectors.yaml
    with open(tv_conf_file, "r") as file:
            config = yaml.safe_load(file)
            config['epoch_divisor'] = epoch_divisor
            config['order'] = order
            print(config)
    with open(tv_conf_file, "w") as file:
        yaml.dump(config, file)
    

    datasets = ["cifar100", "dtd", "eurosat", "gtsrb", "mnist", "resisc45", "svhn"]
    for dataset in datasets: # modify the dataset hyperparameter in config
        with open(yaml_file, "r") as file:
            config = yaml.safe_load(file)
            config['defaults'][0]['dataset'] = dataset
            print(config)

        with open(yaml_file, "w") as file:
            yaml.dump(config, file)

        subprocess.run(["python", "src/scripts/finetune.py"], check=True)

        print(f"\n\n\nExperiment for dataset {dataset} completed.\n\n\n")

    subprocess.run(["python", "src/scripts/evaluate.py"], check=True)
    