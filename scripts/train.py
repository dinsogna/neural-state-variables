import subprocess
import sys

exit_code = subprocess.call('./encoder_decoder_64_train.sh circular_motion 2')
print(exit_code)

# dataset_name = sys.argv[1]
# gpu_no = sys.argv[2]

# print(dataset_name)
# print(gpu_no)

# program_list = [f'./scripts/encoder_decoder_64_train.sh {dataset_name} {gpu_no}', 
#                 f'.scripts/encoder_decoder_train.sh {dataset_name} {gpu_no}', 
#                 f'.scripts/encoder_decoder_64_eval.sh {dataset_name} {gpu_no}', 
#                 f'.scripts/encoder_decoder_eval.sh {dataset_name} {gpu_no}'
#                 ]

# for program in program_list:
#     print("Program:", program)
#     subprocess.call(program)
#     print("Finished:" + program)