import numpy as np
import argparse
import os

parser = argparse.ArgumentParser(description="Mix data with labels for classifier training")
parser.add_argument("--inputOne", required=True, help="Path to the input prerpocessed npy file.")
parser.add_argument("--inputTwo", required=True, help="Path to the input prerpocessed npy file.")
parser.add_argument("--output", required=True, help="Directory to save the result.")
parser.add_argument("--name", required=True, help="Name to save output")
parser.add_argument("--labeled", required=True, help="Labeling")
parser.add_argument("--parts",required=True,help="Number of parts")



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

def mix_npy_files_without(file1_path, file2_path, parts, output_data_path=None):
    """
    Load two .npy files, combine them, shuffle the data, divide into parts, and optionally save.
    
    Args:
        file1_path (str): Path to first .npy file
        file2_path (str): Path to second .npy file
        parts (int): Number of divisions for the combined data
        output_data_path (str, optional): Base path to save the divided data parts
    
    Returns:
        list of numpy.ndarray: List of divided and shuffled data parts
    """
    data1 = np.load(file1_path,mmap_mode="r")
    print(f'Length of first is {len(data1)}')
    data2 = np.load(file2_path,mmap_mode="r")
    print(f'Length of second is {len(data2)}')

    combined_data = np.concatenate([data1, data2], axis=0)
    np.random.shuffle(combined_data)
    print(f"Length of combined is {len(combined_data)}")
    
    # Divide the data into `parts` subsets
    divided_data = np.array_split(combined_data, parts)
    
    if output_data_path:
        for i, part in enumerate(divided_data):
            part_path = f"{output_data_path}_part{i}.npy"
            np.save(part_path, part)
            print(f"Part {i+1} saved to {part_path}")
    
    return divided_data


output_data_path = os.path.join(args.output, f"{args.name}_data.npy")
output_label_path = os.path.join(args.output, f"{args.name}_labels.npy")
if int(args.labeled) == 1:
    mix_npy_files_with_labels(args.inputOne, args.inputTwo, output_data_path, output_label_path)
else:    
    mix_npy_files_without(args.inputOne, args.inputTwo,int(args.parts), output_data_path)
