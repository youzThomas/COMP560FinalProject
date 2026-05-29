# Data

Large dataset files are not tracked in Git.

Expected local paths:

```text
data/raw/dataset_a.mat
data/processed/labels_with_tool_class.csv
data/processed/X_train.hdf5
data/processed/y_train.hdf5
data/processed/X_val.hdf5
data/processed/y_val.hdf5
data/processed/X_test.hdf5
data/processed/y_test.hdf5
```

The default config reads `data/raw/dataset_a.mat` and
`data/processed/labels_with_tool_class.csv`. Processed HDF5 files are local
artifacts used by the existing training/evaluation workflow.
