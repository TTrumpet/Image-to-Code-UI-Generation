from datasets import load_dataset

def print_hf_examples(ds, split_name, num_examples=3):
    for i, example in enumerate(ds[split_name]):
        print(f"Example {i}:")
        for key, value in example.items():
            print(f"  {key}: {value}")
        if i >= num_examples - 1:
            break

def iterate_entire_dataset(ds, split_name):
    for i, example in enumerate(ds[split_name]):
        # Process each example as needed
        print(f"Example {i}:")
        for key, value in example.items():
            print(f"  {key}: {value}")

if __name__ == "__main__":
    dataset_name = "MBZUAI/Web2Code"
    ds = load_dataset(dataset_name)
    split_name = list(ds.keys())[0]
    print(f"Available splits: {list(ds.keys())}")
    print(f"Length of '{split_name}' split: {len(ds[split_name])}")

    print_hf_examples(ds, split_name, num_examples=3)
    # To iterate through the whole dataset, uncomment the following line:
    # iterate_entire_dataset(ds, split_name)

    # NOTE: It seems like loading the dataset from Hugging Face only allows access to 100 rows.
    # Try dataloader.py file and downloading the dataset manually for full access.