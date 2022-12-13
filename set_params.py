import os
import yaml

# define a custom representer for yaml strings
# Define a quoted class, which uses style = '"' and add representer to yaml
class quoted(str):
    pass

def quoted_presenter(dumper, data):
    return dumper.represent_scalar('tag:yaml.org,2002:str', data, style="'")

yaml.add_representer(quoted, quoted_presenter)

datasets = {
        # 0: "test",
        1: "circular_motion",
        2: "single_pendulum",
        3: "double_pendulum",
        4: "elastic_pendulum",
        5: "swingstick_magnetic",
        6: "swingstick_non_magnetic",
        7: "reaction_diffusion",
        8: "fire",
        9: "air_dancer",
        10: "lava_lamp",
    }

params = {
        1: "epochs",
        2: "lr",
        3: "if_cuda",
        4: "gamma",
        5: "train_batch",
        6: "val_batch",
        7: "test_batch",
        8: "num_workers",
        9: "data_filepath",
        10: "num_gpus",
        11: "lambda_schedule"
}

print("=====================")
print("DATASETS")
print("=====================")
for key, value in datasets.items():
    print(f"{key}: {value}")
print()

while True:
    try:
        choice = int(input("Select a dataset (0 to exit): "))
        if choice < 0 or choice > 10:
            print("Not a valid selection. Please try again.")
        else:
            break
    except ValueError:
        print("Please enter an integer.")

if choice == 0:
    print("Goodbye!")
    exit()
dataset_name = datasets[choice]
    
print()
print("=====================")    
print("HYPERPARAMETERS:")
print("=====================")
for key, value in params.items():
    print(f"{key}: {value}")
print()

while True:
    try:
        choice = int(input("Select a hyperparameter to edit (select 0 to exit): "))
        if choice < 0 or choice > 11:
            print("Not a valid selection. Please try again.")
        else:
            break
    except ValueError:
        print("Please enter an integer.")

if choice == 0:
    print("Goodbye!")
    exit()
print()
param_name = params[choice]

while True:
    try:
        new_param_value = input(f"Select a new value for {param_name}: ")
        print("new param value:", new_param_value)
        print("new param type:", type(new_param_value))
        print("split:", new_param_value[1:-1].split(","))
        if param_name in ['data_filepath']:
            break
        if param_name in ['if_cuda']:
            if new_param_value in ['True', 'true', 'T', 't']:
                new_param_value = True
            else:
                new_param_value = False
            break
        elif param_name in ['train_batch', 'val_batch', 'test_batch', 'num_workers', 'num_gpus', 'epochs']:
            new_param_value = int(new_param_value)
            break
        elif param_name in ['lr', 'gamma']:
            new_param_value = float(new_param_value)
            break
        elif param_name in ['lambda_schedule']:
            out = []
            new_param_value = new_param_value[1:-1].split(",")
            for val in new_param_value:
                out.append(float(val))
            new_param_value = out
            print("new param value after processing:", new_param_value)
            print("type:", type(new_param_value))
            break
    except ValueError:
        print("Invalid value try again. ")

# update_param(dataset_name, param_name, new_param_value)
path = f"configs/{dataset_name}"

for dir in os.listdir(path):
    path_dir = path + "/" + dir
    for f in os.listdir(path_dir):
        path_yaml = path_dir + "/" + f
        with open(path_yaml) as yf:
            yaml_dict = yaml.load(yf, Loader=yaml.FullLoader)
        yaml_dict[param_name] = new_param_value
        for key, val in yaml_dict.items():
            if type(val) == str:
                yaml_dict[key] = quoted(val)
        with open(path_yaml, 'w') as yf:
            yaml.dump(yaml_dict, yf, sort_keys=False, default_flow_style=None)
print("Success!")


# if __name__ == '__main__':
#    main()
