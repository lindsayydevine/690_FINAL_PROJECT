import h5py
import numpy as np

# AccLabel is acceleration filtered and frequency aligned
with h5py.File('./preprocessed_data/Data_AccLabel_P034.h5', 'r') as f:
    # List all the "folders" (datasets) inside
    print("Keys in HDF5:", list(f.keys()))
    
    # Peek at the labels
    labels = f['window_label'][:]
    print("\nFirst 10 labels:", labels[:10])
    print("Unique labels in this file:", np.unique(labels))
    
    # Check the shape of your movement elements
    acc_data = f['window_acc_filt_gravity'][:]
    print("\nShape of Movement Elements:", acc_data.shape)
    # (Number of windows, Pad size, Features)


# MeLabel is movement elements based on Zero Crossings
with h5py.File('./preprocessed_data/Data_MeLabel_P034.h5', 'r') as f:
    # List all the "folders" (datasets) inside
    print("Keys in HDF5:", list(f.keys()))
    
    # Peek at the labels
    labels = f['window_label'][:]
    print("\nFirst 10 labels:", labels[:10])
    print("Unique labels in this file:", np.unique(labels))
    
    # Check the shape of your movement elements
    me_data = f['x_acc_filt'][:]
    print("\nShape of Movement Elements:", me_data.shape)
    # (Number of windows, Pad size, Features)