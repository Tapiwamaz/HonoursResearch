import numpy as np
import argparse
import os

parser = argparse.ArgumentParser(description="Mix data with labels for classifier training")
parser.add_argument("--inputOne", required=True, help="Path to the input prerpocessed npy file.")
parser.add_argument("--inputTwo", required=True, help="Path to the input prerpocessed npy file.")
parser.add_argument("--output", required=True, help="Directory to save the result.")
parser.add_argument("--name", required=True, help="Name to save output")


args = parser.parse_args()

def mix_npy_files_with_labels(file1_path, file2_path, output_data_path=None, output_label_path=None):
    data1 = np.load(file1_path)
    data2 = np.load(file2_path)
    labels1 = np.zeros(len(data1), dtype=np.int64)
    labels2 = np.ones(len(data2), dtype=np.int64)
    
    combined_data = np.concatenate([data1, data2], axis=0)
    combined_labels = np.concatenate([labels1, labels2], axis=0)
    
    # Shuffle together
    indices = np.arange(len(combined_data))
    np.random.shuffle(indices)
    combined_data = combined_data[indices]
    combined_labels = combined_labels[indices]
    
    if output_data_path:
        np.save(output_data_path, combined_data)
        print(f"Data saved to {output_data_path}")
    if output_label_path:
        np.save(output_label_path, combined_labels)
        print(f"Labels saved to {output_label_path}")
    return combined_data, combined_labels

def mix_npy_files_without(file1_path, file2_path, output_data_path=None):
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
    
    if output_data_path:
        np.save(output_data_path, combined_data)
        print(f"File save to {output_data_path}")
   
    return combined_data


output_data_path = os.path.join(args.output, f"{args.name}_data.npy")
output_label_path = os.path.join(args.output, f"{args.name}_labels.npy")
# mix_npy_files_with_labels(args.inputOne, args.inputTwo, output_data_path, output_label_path)
mix_npy_files_without(args.inputOne, args.inputTwo, output_data_path)
