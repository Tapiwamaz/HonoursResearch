import numpy as np
import argparse
import os

parser = argparse.ArgumentParser(description="MIX DATA for pretraining")
parser.add_argument("--inputOne", required=True, help="Path to the input prerpocessed npy file.")
parser.add_argument("--inputTwo", required=True, help="Path to the input prerpocessed npy file.")
parser.add_argument("--output", required=True, help="Directory to save the result.")
parser.add_argument("--name", required=True, help="Name to save output")


args = parser.parse_args()

def mix_npy_files(file1_path, file2_path, output_path=None):
    """
    Load two .npy files, combine them, shuffle the data, and optionally save.
    
    Args:
        file1_path (str): Path to first .npy file
        file2_path (str): Path to second .npy file
        output_path (str, optional): Path to save the mixed data
    
    Returns:
        numpy.ndarray: Combined and shuffled data
    """
    data1 = np.load(file1_path)
    print(f'length of first is {len(data1)}')
    data2 = np.load(file2_path)
    print(f'length of second is {len(data2)}')

    combined_data = np.concatenate([data1, data2], axis=0)
    np.random.shuffle(combined_data)
    print(f"Length of combined is {len(combined_data)}")
    
    if output_path:
        np.save(output_path, combined_data)
        print(f"File save to {output_path}")
    

# Example usage:
output_path = os.path.join(args.output, f"{args.name}.npy")
mix_npy_files(file1_path=args.inputOne, file2_path=args.inputTwo, output_path=output_path)
#print(f'Length of mixed is {len(np.load(args.inputOne))}')
