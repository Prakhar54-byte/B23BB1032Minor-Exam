from datasets import load_dataset

ds = load_dataset("Chiranjeev007/CIFAR-10_Subset")
print(ds)
# DatasetDict({
#   train:      Dataset(num_rows: 5000),
#   validation: Dataset(num_rows: 500),
#   test:       Dataset(num_rows: 1000)
# })

sample = ds["train"][0]
sample["image"]   # PIL Image 32×32 RGB
sample["label"]   # int 0–9
