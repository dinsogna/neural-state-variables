import os
import yaml

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
        1: "lr",
        2: "if_cuda",
        3: "gamma",
        4: "train_batch",
        5: "val_batch",
        6: "test_batch",
        7: "num_workers",
        8: "data_filepath",
        9: "num_gpus",
        10: "epochs"
}

def update_param(dataset_name, param_name, new_param_value):
    path = f"configs/{dataset_name}"
    # new_param_value = verify_param(param_name, new_value)

    for dir in os.listdir(path):
        path_dir = path + "/" + dir
        for f in os.listdir(path_dir):
            path_yaml = path_dir + "/" + f
            with open(path_yaml) as yf:
                yaml_dict = yaml.load(yf, Loader=yaml.FullLoader)
            yaml_dict[param_name] = new_param_value
            with open(path_yaml, 'w') as yf:
                yaml.dump(yaml_dict, yf, sort_keys=False, default_flow_style=None)



def main():

    print("DATASETS")
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
    print("HYPERPARAMETERS:")
    for key, value in params.items():
        print(f"{key}: {value}")

    while True:
        try:
            choice = int(input("Select a hyperparameter to edit (select 0 to exit): "))
            if choice < 0 or choice > 10:
                print("Not a valid selection. Please try again.")
            else:
                break
        except ValueError:
            print("Please enter an integer.")

    if choice == 0:
        print("Goodbye!")
        exit()
    param_name = params[choice]

    while True:
        try:
            new_param_value = input(f"Select a new value for {param_name}: ")
            if param_name in ['data_filepath']:
                break
            elif param_name in ['lr', 'if_cuda', 'gamma', 'train_batch', 'val_batch', 'test_batch', 'num_workers', 'num_gpus', 'epochs']:
                new_param_value = int(new_param_value)
                break
        except ValueError:
            print("Invalid value try again. ")

    update_param(dataset_name, param_name, new_param_value)
    print("Success!")


if __name__ == '__main__':
   main()
