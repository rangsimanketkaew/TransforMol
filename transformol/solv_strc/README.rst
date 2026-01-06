============================================
Molecular Geometry Prediction Neural Network
============================================

A neural network model for predicting molecular geometries in solvent phase given gas phase geometries and solvent dielectric constants.

Overview
========

1. Load geometry from HDF5 files, create and split dataset
2. Calculate molecular features (global descriptors)
3. Model (MLP)
4. Training function
5. Hyperparameter optimization
6. Prediction
7. Analysis and visualization

Requirements
============

::

    torch >= 1.9.0
    numpy >= 1.21.0
    pandas >= 1.3.0
    h5py >= 3.1.0
    scikit-learn >= 0.24.0
    matplotlib >= 3.4.0

Usage
=====

Step 1: Generate Dataset
-------------------------

Generate a dataset from your HDF5 files:

.. code-block:: bash

    # Generate full dataset
    python generate_dataset.py --gas AQM-gas.hdf5 --sol AQM-sol.hdf5 --output molecule_dataset.csv

    # Or use `--n-samples N` to generate a sample dataset for testing (N molecules)

Step 2: Train the Neural Network
---------------------------------

.. code-block:: bash

    # training
    python train.py --data-path molecule_dataset.csv --epochs 1000

    # training with hyperparam optimization
    python train.py --data-path molecule_dataset.csv --epochs 1000 --optimize --n-trials 10

Using Pre-trained Model
=======================

.. code-block:: python

    # Load best model
    checkpoint = torch.load('best_model.pt')
    model.load_state_dict(checkpoint['model_state_dict'])

    # Make prediction (example)
    gas_xyz = np.array([[0, 0, 0], [1.5, 0, 0], [0, 1.5, 0]])
    atom_nums = np.array([6, 1, 1])
    predicted_sol_xyz = predict_geometry(
        model, gas_xyz, atom_nums, dielectric=78.4,
        scaler=data['scaler']
    )

    print("Predicted solvent geometry:")
    print(predicted_sol_xyz)

Output Metrics
==============

* **Training/Validation Loss**: MSE between predicted and true coordinates
* **Test Metrics**:

  * MSE (Mean Squared Error)
  * RMSE (Root Mean Squared Error)
  * MAE (Mean Absolute Error)
  * Per-coordinate error analysis
  * Per-molecule RMSE distribution
