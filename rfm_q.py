# # q_rfm.py
# import numpy as np
# import torch
# from numpy.linalg import solve
# from tqdm import tqdm
# import hickle
# import time


# # Import Qiskit libraries for the quantum kernel


# #from qiskit.utils.quantum_instance import QuantumInstance  ###
# import qiskit
# from qiskit import QuantumCircuit
# from qiskit_aer import Aer  # Use qiskit_aer instead of qiskit.providers.aer
# from qiskit.primitives import Sampler
# #from qiskit.primitives.fidelities import ComputeUncompute

# from qiskit.circuit.library import ZZFeatureMap
# # from qiskit_machine_learning import QuantumKernel ###
# from qiskit_machine_learning.kernels import FidelityQuantumKernel
# from qiskit.primitives import Estimator
# from qiskit.quantum_info import state_fidelity
# # def encode_features(X):
# #     """
# #     Scales each column of X to [0, 1], then multiplies by pi.
# #     X: NumPy array of shape (n_samples, d).
# #     Returns scaled_X: the same shape, but scaled to [0, pi].
# #     """
# #     # Avoid modifying the original array
# #     X = X.copy()
    
# #     # For numerical stability, handle columns where max==min
# #     for j in range(X.shape[1]):
# #         col_min = X[:, j].min()
# #         col_max = X[:, j].max()
# #         if abs(col_max - col_min) < 1e-12:
# #             # If this feature is essentially constant, just set it to 0
# #             X[:, j] = 0.0
# #         else:
# #             X[:, j] = (X[:, j] - col_min) / (col_max - col_min)
    
# #     # Now scale [0,1] --> [0, pi]
# #     X *= np.pi
# #     return X


# # def encode_features(X):
# #     """
# #     Scales each column of X to [0, 1], then multiplies by pi.
# #     X: NumPy array of shape (n_samples, d).
# #     Returns scaled_X: the same shape, but scaled to [0, pi].
# #     """
# #     X = X.copy()
# #     for j in range(X.shape[1]):
# #         col_min = X[:, j].min()
# #         col_max = X[:, j].max()
# #         if abs(col_max - col_min) < 1e-12:
# #             X[:, j] = 0.0
# #         else:
# #             X[:, j] = (X[:, j] - col_min) / (col_max - col_min)
# #     # Now scale [0,1] --> [0, pi]
# #     X *= np.pi
# #     return X

# def encode_features(X):
#     """
#     Scales each column of X to [0, 1], then multiplies by pi.
#     X: NumPy array of shape (n_samples, d).
#     Returns scaled_X: the same shape, but scaled to [0, pi].
#     """
#     X = X.copy()
#     for j in range(X.shape[1]):
#         col_min = X[:, j].min()
#         col_max = X[:, j].max()
#         if abs(col_max - col_min) < 1e-12:
#             X[:, j] = 0.0
#         else:
#             X[:, j] = (X[:, j] - col_min) / (col_max - col_min)
#     # Now scale [0,1] --> [0, pi]
#     X *= np.pi
#     return X

#  # ✅ Replace StateFidelity
# # def quantum_kernel_matrix(X1, X2, q_kernel):
# #     """
# #     Compute the quantum kernel matrix for inputs X1 and X2.
# #     X1, X2: torch.Tensor or np.ndarray of shape (n_samples, n_features)
# #     q_kernel: an instance of Qiskit's QuantumKernel.
# #     Returns:
# #     ImportError: cannot import name 'Aer' from 'qiskit' (/data/yi/minico
# #         K: np.ndarray of shape (n_samples_X1, n_samples_X2)
# #     """
# #     if isinstance(X1, torch.Tensor):
# #         X1 = X1.cpu().numpy()
# #     if isinstance(X2, torch.Tensor):
# #         X2 = X2.cpu().numpy()
    
# #     # # Evaluate the quantum kernel
# #     # K = q_kernel.evaluate(x_vec=X1, y_vec=X2)
# #     # print("[DEBUG] quantum_kernel_matrix: shape =", K.shape)
# #     # return K
    
    
# #         # Encode (scale) data into [0, pi], if desired
# #     if do_encode:
# #         X1 = encode_features(X1)
# #         X2 = encode_features(X2)
        
        
# #     print("[DEBUG] Evaluating Quantum Kernel...")  # ✅ Add this line
# #     start_time = time.time()
# #     K = q_kernel.evaluate(x_vec=X1, y_vec=X2)  # <- Suspected slow part
# #     end_time = time.time()
# #     print("[DEBUG] Quantum Kernel evaluated successfully!")  # ✅ Add this line
# #     print("[DEBUG] quantum_kernel_matrix: shape =", K.shape)
# #     return K

# # def quantum_kernel_matrix(X1, X2, q_kernel, do_encode=True):
# #     """
# #     Compute the quantum kernel matrix for inputs X1 and X2.
    
# #     X1, X2: torch.Tensor or np.ndarray of shape (n_samples, n_features)
# #     q_kernel: an instance of Qiskit's QuantumKernel.
# #     do_encode: whether to scale/encode data before evaluating.
    
# #     Returns:
# #         K: np.ndarray of shape (n_samples_X1, n_samples_X2)
# #     """
# #     # Convert to NumPy if Torch
# #     if isinstance(X1, torch.Tensor):
# #         X1 = X1.cpu().numpy()
# #     if isinstance(X2, torch.Tensor):
# #         X2 = X2.cpu().numpy()
    
    
# #     if do_encode:
# #         X1 = encode_features(X1)
# #         X2 = encode_features(X2)

# #     # If M is provided, transform data by sqrt(M)
# #     if M is not None:
# #         # Compute sqrt(M) (Cholesky or other stable sqrt)
# #         sqrtM = np.real_if_close(np.linalg.cholesky(M))
# #         X1 = X1 @ sqrtM
# #         X2 = X2 @ sqrtM
# #     # Encode (scale) data into [0, pi], if desired
# #     # 1) Transform each vector x -> sqrt(M) x in numpy

# #     sqrtM = np.real_if_close(np.linalg.cholesky(M)) 
# #     print("[DEBUG] Evaluating Quantum Kernel...")
# #     start_time = time.time()
# #     # K = q_kernel.evaluate(x_vec=X1, y_vec=X2)  # Fidelity kernel
# #     X1_transformed = (X1 @ sqrtM)  # or np.dot(X1, sqrtM)
# #     X2_transformed = (X2 @ sqrtM)

# #     # 2) Then pass those to the quantum kernel
# #     K = q_kernel.evaluate(x_vec=X1_transformed, y_vec=X2_transformed)
    
# #     end_time = time.time()
# #     print("[DEBUG] Quantum Kernel evaluated successfully! Shape =", K.shape)
# #     print("[DEBUG] Time = %.2f seconds" % (end_time - start_time))
# #     return K

# def quantum_kernel_matrix(X1, X2, q_kernel, M=None, do_encode=True):
#     """
#     Compute the quantum kernel matrix for inputs X1 and X2, 
#     optionally transforming each vector x -> sqrt(M) * x first.

#     Args:
#         X1, X2: torch.Tensor or np.ndarray of shape (n_samples, n_features)
#         q_kernel: an instance of Qiskit's FidelityQuantumKernel (or similar)
#         M: np.ndarray (d x d), the matrix from RFM. If None, no transform.
#         do_encode: whether to scale/encode data [0,1]->[0, pi] before the quantum feature map.

#     Returns:
#         K: np.ndarray of shape (n_samples_X1, n_samples_X2)
#     """
#     # Convert to NumPy
#     if isinstance(X1, torch.Tensor):
#         X1 = X1.cpu().numpy()
#     if isinstance(X2, torch.Tensor):
#         X2 = X2.cpu().numpy()

#     # Optionally encode [0,1] -> [0, pi]
#     if do_encode:
#         X1 = encode_features(X1)
#         X2 = encode_features(X2)

#     # If M is provided, transform data by sqrt(M)
#     if M is not None:
#         # Compute sqrt(M) (Cholesky or other stable sqrt)
#         sqrtM = np.real_if_close(np.linalg.cholesky(M))
#         X1 = X1 @ sqrtM
#         X2 = X2 @ sqrtM

#     print("[DEBUG] Evaluating Quantum Kernel...")
#     start_time = time.time()
#     # Evaluate fidelity-based kernel
#     K = q_kernel.evaluate(x_vec=X1, y_vec=X2)
#     end_time = time.time()

#     print("[DEBUG] Quantum Kernel evaluated successfully! Shape =", K.shape)
#     print("[DEBUG] Time = %.2f seconds" % (end_time - start_time))
#     return K

# def get_grads(X, sol, L, M, q_kernel, batch_size=2, do_encode=True):
#     """
#     Quantum-based gradient update that uses the quantum kernel in place of a classical kernel,
#     but also transforms the data by sqrt(M) when computing the kernel matrix.

#     Args:
#         X: training data (torch.Tensor or np.ndarray)
#         sol: solution vector from the last solve(...) call
#         L: some scalar parameter (like 'bandwidth' or 'scale')
#         M: the RFM matrix we are updating
#         q_kernel: FidelityQuantumKernel
#         batch_size: mini-batch size for summing gradient outer products
#         do_encode: whether to scale data to [0, pi] as well

#     Returns:
#         M_new: updated version of M
#     """
#     print("[DEBUG] Entering get_grads...")

#     # We only sample up to 1000 points for gradient update to reduce overhead
#     num_samples = min(len(X), 1000)
#     indices = np.random.choice(len(X), size=num_samples, replace=False)
#     x = X[indices, :]

#     # NxM quantum kernel: transform by sqrt(M) inside quantum_kernel_matrix
#     print("[DEBUG] Computing quantum kernel matrix for gradients...")
#     K = quantum_kernel_matrix(X, x, q_kernel, M=M, do_encode=do_encode)
#     print("[DEBUG] Quantum kernel matrix computed for gradient shape:", K.shape)

#     # Convert sol to torch
#     a1 = torch.from_numpy(sol.T).float()
#     n, d = X.shape
#     n, c = a1.shape
#     m, d = x.shape

#     # step1
#     a1 = a1.reshape(n, c, 1)
#     X1 = torch.from_numpy(X).float() @ torch.from_numpy(M).float()  # X @ M
#     X1 = X1.reshape(n, 1, d)
#     step1 = a1 @ X1
#     del a1, X1
#     step1 = step1.reshape(-1, c*d)

#     # step2
#     step2 = torch.from_numpy(K).float().T @ step1
#     del step1
#     step2 = step2.reshape(-1, c, d)

#     # step3
#     a2 = torch.from_numpy(sol).float()
#     step3 = (a2 @ torch.from_numpy(K).float()).T
#     del K, a2
#     step3 = step3.reshape(m, c, 1)

#     x1 = torch.from_numpy(x).float() @ torch.from_numpy(M).float()  # x @ M
#     x1 = x1.reshape(m, 1, d)
#     step3 = step3 @ x1

#     # Final gradient G
#     G = (step2 - step3) * -1.0 / L

#     # Accumulate outer products into M_new
#     M_new = 0.0
#     bs = batch_size
#     batches = torch.split(G, bs)
#     for i in range(len(batches)):
#         grad = batches[i]
#         gradT = torch.transpose(grad, 1, 2)
#         M_new += torch.sum(gradT @ grad, dim=0).cpu()
#         del grad, gradT

#     M_new /= len(G)
#     M_new = M_new.numpy()
#     return M_new
# # def get_grads(X, sol, L, P, q_kernel, batch_size=2):
# #     """
# #     This function mimics the classical gradient-based update.
# #     In this example, we keep the structure of your classical get_grads,
# #     but note that quantum kernels typically do not update an auxiliary matrix.
# #     """
# #     print("[DEBUG] Entering get_grads...")
# #     print("[DEBUG] Checking dataset size: ", len(X))

# #     #M = 0.0

# #     #num_samples = 20000
# #     num_samples = min(len(X), 1000) 
# #     indices = np.random.randint(len(X), size=num_samples) if len(X) > 20000 else np.arange(len(X))
# #     x = X[indices, :]

# #     # Use the quantum kernel for computing K_Q(X, x)
# #     # K = quantum_kernel_matrix(X, x, q_kernel)
# #     print("[DEBUG] Computing quantum kernel matrix for gradients...")
# #     K = quantum_kernel_matrix(X, x, q_kernel)
# #     print("[DEBUG] Quantum kernel matrix computed for gradients!")

# #     # Convert sol to torch tensor
# #     a1 = torch.from_numpy(sol.T).float()
# #     n, d = X.shape
# #     n, c = a1.shape
# #     m, d = x.shape

# #     a1 = a1.reshape(n, c, 1)
# #     X1 = (X @ P).reshape(n, 1, d)
# #     step1 = a1 @ X1
# #     del a1, X1
# #     step1 = step1.reshape(-1, c * d)

# #     # step2 = torch.from_numpy(K).T @ step1
# #     step2 = torch.from_numpy(K).float().T @ step1
# #     del step1
# #     step2 = step2.reshape(-1, c, d)

# #     a2 = torch.from_numpy(sol).float()
# #     # step3 = (a2 @ torch.from_numpy(K)).T
# #     step3 = (a2 @ torch.from_numpy(K).float()).T
# #     del K, a2

# #     step3 = step3.reshape(m, c, 1)
# #     x1 = (x @ P).reshape(m, 1, d)
# #     step3 = step3 @ x1

# #     G = (step2 - step3) * -1 / L

# #     M = 0.0
# #     bs = batch_size
# #     batches = torch.split(G, bs)
# #     for i in tqdm(range(len(batches))):
# #         print(f"[DEBUG] Processing batch {i+1}/{len(batches)}")
# #         grad = batches[i]
# #         gradT = torch.transpose(grad, 1, 2)
# #         M += torch.sum(gradT @ grad, dim=0).cpu()
# #         del grad, gradT
# #     M /= len(G)
# #     M = M.numpy()
# #     return M
# def get_grads(X, sol, L, P, q_kernel, batch_size=2):
#     """
#     This function mimics the classical gradient-based update,
#     but uses a quantum kernel in place of the classical kernel.
#     """
#     print("[DEBUG] Entering get_grads...")

#     # We only sample up to 1000 points for the update:
#     num_samples = min(len(X), 1000)
#     indices = np.random.choice(len(X), size=num_samples, replace=False)
#     x = X[indices, :]
    
#     # Compute the NxM quantum kernel (where N=len(X), M=num_samples)
#     print("[DEBUG] Computing quantum kernel matrix for gradients...")
#     K = quantum_kernel_matrix(X, x, q_kernel, do_encode=True)  # do_encode can be True or False
#     print("[DEBUG] Quantum kernel matrix computed for gradient shape:", K.shape)

#     # Convert sol to torch
#     a1 = torch.from_numpy(sol.T).float()
#     n, d = X.shape
#     n, c = a1.shape
#     m, d = x.shape

#     a1 = a1.reshape(n, c, 1)
#     X1 = (X @ P).reshape(n, 1, d)
#     step1 = a1 @ X1
#     del a1, X1
#     step1 = step1.reshape(-1, c*d)

#     step2 = torch.from_numpy(K).float().T @ step1
#     del step1
#     step2 = step2.reshape(-1, c, d)

#     a2 = torch.from_numpy(sol).float()
#     step3 = (a2 @ torch.from_numpy(K).float()).T
#     del K, a2

#     step3 = step3.reshape(m, c, 1)
#     x1 = (x @ P).reshape(m, 1, d)
#     step3 = step3 @ x1

#     # Final gradient G
#     G = (step2 - step3) * -1 / L

#     M = 0.0
#     bs = batch_size
#     batches = torch.split(G, bs)
#     for i in range(len(batches)):
#         grad = batches[i]
#         gradT = torch.transpose(grad, 1, 2)
#         M += torch.sum(gradT @ grad, dim=0).cpu()
#         del grad, gradT

#     M /= len(G)
#     M = M.numpy()
#     return M

# # def get_data(loader):
# #     X = []
# #     y = []
# #     for idx, batch in enumerate(loader):
# #         inputs, labels = batch
# #         X.append(inputs)
# #         y.append(labels)
# #     return torch.cat(X, dim=0), torch.cat(y, dim=0)

# # def get_data(loader):
# #     X = []
# #     y = []
# #     for idx, batch in enumerate(loader):
# #         inputs, labels = batch
# #         X.append(inputs)
# #         y.append(labels)
# #     X = torch.cat(X, dim=0)
# #     y = torch.cat(y, dim=0)
    
# #     #  Take only the first 500 samples for debugging
# #     X = X[:500]
# #     y = y[:500]
    
# #     return X, y

# def get_data(loader):
#     """
#     Gather all data from a PyTorch DataLoader into single tensors (X, y).
#     For debugging, limit to 500 samples.
#     """
#     X_list, y_list = [], []
#     for idx, batch in enumerate(loader):
#         inputs, labels = batch
#         X_list.append(inputs)
#         y_list.append(labels)
#     X = torch.cat(X_list, dim=0)
#     y = torch.cat(y_list, dim=0)
    
#     #  Take only the first 500 samples for debugging
#     X = X[:500]
#     y = y[:500]
    
#     return X, y


# # def q_rfm(train_loader, test_loader,
# #           iters=3, name=None, batch_size=2, reg=1e-3,
# #           train_acc=False, loader=True, classif=True):
# #     print("[DEBUG] Entered q_rfm function...")
# #     """
# #     Quantum version of the Recursive Feature Machine.
# #     This function replaces the classical laplace_kernel_M with a quantum kernel.
# #     """
# #     # Create a quantum kernel using Qiskit's ZZFeatureMap as an example
# #     # Set feature dimension based on the data
# #     X_train_dummy, _ = get_data(train_loader) if loader else train_loader
# #     feature_dim = X_train_dummy.shape[1]
# #     feature_map = ZZFeatureMap(feature_dimension=feature_dim, reps=1, entanglement="full")
# #     print("[DEBUG] Initializing Aer backend...")
# #     backend = Aer.get_backend("aer_simulator_statevector")  # Match your working backends
# #     print("[DEBUG] Aer backend initialized successfully")


# #     print("[DEBUG] Creating FidelityQuantumKernel...")
# #     q_kernel = FidelityQuantumKernel(feature_map=feature_map)
# #     print("[DEBUG] FidelityQuantumKernel created successfully")

# #     X_test = np.random.rand(10, 2)  # Small dataset for quick testing
# #     K_test = q_kernel.evaluate(x_vec=X_test, y_vec=X_test)
# #     print("[DEBUG] Kernel Evaluation Success! Matrix shape:", K_test.shape)

# #     L = 10

# #     if loader:
# #         print("[DEBUG] Loaders provided")
# #         X_train, y_train = get_data(train_loader)
# #         X_test, y_test = get_data(test_loader)
# #     else:
# #         print("[DEBUG] Loaders not used, loading manually")
# #         X_train, y_train = train_loader
# #         X_test, y_test = test_loader
# #         X_train = torch.from_numpy(X_train).float()
# #         X_test = torch.from_numpy(X_test).float()
# #         y_train = torch.from_numpy(y_train).float()
# #         y_test = torch.from_numpy(y_test).float()
# #     print(f"[DEBUG] X_train shape: {X_train.shape}, y_train shape: {y_train.shape}")
# #     print(f"[DEBUG] X_test shape: {X_test.shape}, y_test shape: {y_test.shape}")
# #     # Update feature dimension for the feature map (if necessary)
# #     feature_dim = X_train.shape[1]
# #     feature_map.feature_dimension = feature_dim

# #     n, d = X_train.shape

# #     # Start with an identity matrix M (if still needed for your recursive update)
# #     M = np.eye(d, dtype='float32')

# #     for i in range(iters):
# #         # Compute the quantum kernel matrix for training data
# #         print(f"[DEBUG] Iteration {i+1}/{iters} started...")
# #         print("[DEBUG] Computing K_train...")
# #         K_train = quantum_kernel_matrix(X_train, X_train, q_kernel)
# #         print(f"[DEBUG] K_train computed, shape: {K_train.shape}")
# #         sol = solve(K_train + reg * np.eye(len(K_train)), y_train).T

# #         if train_acc:
# #             preds = (sol @ K_train).T
# #             y_pred = torch.from_numpy(preds)
# #             preds = torch.argmax(y_pred, dim=-1)
# #             labels = torch.argmax(y_train, dim=-1)
# #             count = torch.sum(labels == preds).numpy()
# #             print("Round " + str(i) + " Train Acc: ", count / len(labels))

# #         K_test = quantum_kernel_matrix(X_train, X_test, q_kernel)
# #         preds = (sol @ K_test).T
# #         print("Round " + str(i) + " MSE: ", np.mean(np.square(preds - y_test.numpy())))

# #         if classif:
# #             y_pred = torch.from_numpy(preds)
# #             preds = torch.argmax(y_pred, dim=-1)
# #             labels = torch.argmax(y_test, dim=-1)
# #             count = torch.sum(labels == preds).numpy()
# #             print("Round " + str(i) + " Acc: ", count / len(labels))

# #         # Update M using the gradient-based procedure (optional)
# #         print("[DEBUG] Calling get_grads...")
# #         M = get_grads(X_train, sol, L, torch.from_numpy(M), q_kernel, batch_size=batch_size)
# #         print("[DEBUG] get_grads completed")
# #         if name is not None:
# #             hickle.dump(M, 'saved_Ms/M_' + name + '_' + str(i) + '.h')

# #     K_train = quantum_kernel_matrix(X_train, X_train, q_kernel)
# #     print("[DEBUG] Solving system of equations...")
# #     sol = solve(K_train + reg * np.eye(len(K_train)), y_train).T
# #     print("[DEBUG] System solved")
# #     K_test = quantum_kernel_matrix(X_train, X_test, q_kernel)
# #     preds = (sol @ K_test).T
# #     mse = np.mean(np.square(preds - y_test.numpy()))
# #     print("Final MSE: ", mse)

# #     if classif:
# #         y_pred = torch.from_numpy(preds)
# #         preds = torch.argmax(y_pred, dim=-1)
# #         labels = torch.argmax(y_test, dim=-1)
# #         count = torch.sum(labels == preds).numpy()
# #         print("Final Acc: ", count / len(labels))
# #     return M, mse
# # def q_rfm(
# #     train_loader, 
# #     test_loader,
# #     iters=3, 
# #     name=None, 
# #     batch_size=2, 
# #     reg=1e-3,
# #     train_acc=False, 
# #     loader=True, 
# #     classif=True
# # ):
# #     print("[DEBUG] Entered q_rfm function...")

# #     # 1) Get raw training data
# #     if loader:
# #         X_train, y_train = get_data(train_loader)
# #         X_test,  y_test  = get_data(test_loader)
# #     else:
# #         # If user passes in arrays
# #         X_train, y_train = train_loader
# #         X_test,  y_test  = test_loader
# #         X_train = torch.from_numpy(X_train).float()
# #         y_train = torch.from_numpy(y_train).float()
# #         X_test  = torch.from_numpy(X_test).float()
# #         y_test  = torch.from_numpy(y_test).float()

# #     print("[DEBUG] X_train shape:", X_train.shape, "y_train shape:", y_train.shape)
# #     print("[DEBUG] X_test shape:",  X_test.shape,  "y_test shape:",  y_test.shape)

# #     # 2) Encode/scale training + test data once up front
# #     #    so we don't do it repeatedly inside the kernel calls.
# #     X_train_np = X_train.cpu().numpy()
# #     X_test_np  = X_test.cpu().numpy()
    
# #     X_train_enc = encode_features(X_train_np)  # shape [N, d], scaled to [0, pi]
# #     X_test_enc  = encode_features(X_test_np)

# #     # 3) Build the quantum kernel
# #     feature_dim = X_train.shape[1]
# #     feature_map = ZZFeatureMap(feature_dimension=feature_dim, reps=1, entanglement="full")
# #     backend = Aer.get_backend("aer_simulator_statevector")
    
# #     # Use FidelityQuantumKernel
# #     q_kernel = FidelityQuantumKernel(feature_map=feature_map)
    
# #     # 4) Initialize M, etc.
# #     n, d = X_train.shape
# #     M = np.eye(d, dtype='float32')
# #     L = 10.0

# #     # 5) Iterations
# #     for i in range(iters):
# #         print(f"[DEBUG] Iteration {i+1}/{iters}")
        
# #         # Compute kernel on (encoded) training data
# #         K_train = q_kernel.evaluate(x_vec=X_train_enc, y_vec=X_train_enc)
# #         sol = solve(K_train + reg * np.eye(len(K_train)), y_train.numpy()).T
        
# #         # Training accuracy if needed
# #         if train_acc:
# #             preds_train = (sol @ K_train).T
# #             y_pred = torch.from_numpy(preds_train)
# #             preds_c = torch.argmax(y_pred, dim=-1)
# #             labels_c = torch.argmax(y_train, dim=-1)
# #             acc_train = (labels_c == preds_c).sum().item() / len(labels_c)
# #             print("Round", i, "Train Acc =", acc_train)

# #         # Test MSE
# #         K_test = q_kernel.evaluate(x_vec=X_train_enc, y_vec=X_test_enc)
# #         preds = (sol @ K_test).T
# #         mse = np.mean((preds - y_test.numpy())**2)
# #         print("Round", i, "MSE =", mse)

# #         # Test classification accuracy
# #         if classif:
# #             y_pred = torch.from_numpy(preds)
# #             preds_c = torch.argmax(y_pred, dim=-1)
# #             labels_c = torch.argmax(y_test, dim=-1)
# #             acc_test = (labels_c == preds_c).sum().item() / len(labels_c)
# #             print("Round", i, "Acc =", acc_test)

# #         # Update M using get_grads
# #         M = get_grads(X_train, sol, L, torch.from_numpy(M), q_kernel, batch_size=batch_size)
# #         if name is not None:
# #             hickle.dump(M, f'saved_Ms/M_{name}_{i}.h')

# #     # Final pass
# #     K_train = q_kernel.evaluate(x_vec=X_train_enc, y_vec=X_train_enc)
# #     sol = solve(K_train + reg * np.eye(len(K_train)), y_train.numpy()).T
# #     K_test = q_kernel.evaluate(x_vec=X_train_enc, y_vec=X_test_enc)
# #     preds = (sol @ K_test).T
# #     mse = np.mean((preds - y_test.numpy())**2)
# #     print("Final MSE:", mse)

# #     if classif:
# #         y_pred = torch.from_numpy(preds)
# #         preds_c = torch.argmax(y_pred, dim=-1)
# #         labels_c = torch.argmax(y_test, dim=-1)
# #         acc_test = (labels_c == preds_c).sum().item() / len(labels_c)
# #         print("Final Acc:", acc_test)

# #     return M, mse

# def q_rfm(
#     train_loader, 
#     test_loader,
#     iters=3, 
#     name=None, 
#     batch_size=2, 
#     reg=1e-3,
#     train_acc=False, 
#     loader=True, 
#     classif=True
# ):
#     """
#     Quantum version of the Recursive Feature Machine (RFM).
#     We incorporate M by transforming data with sqrt(M) inside the quantum kernel.
#     """
#     print("[DEBUG] Entered q_rfm function...")

#     # 1) Get raw training data
#     if loader:
#         X_train, y_train = get_data(train_loader)
#         X_test,  y_test  = get_data(test_loader)
#     else:
#         # If user passes in arrays
#         X_train, y_train = train_loader
#         X_test,  y_test  = test_loader
#         X_train = torch.from_numpy(X_train).float()
#         y_train = torch.from_numpy(y_train).float()
#         X_test  = torch.from_numpy(X_test).float()
#         y_test  = torch.from_numpy(y_test).float()

#     print("[DEBUG] X_train shape:", X_train.shape, "y_train shape:", y_train.shape)
#     print("[DEBUG] X_test shape:",  X_test.shape,  "y_test shape:",  y_test.shape)

#     # 2) Build the quantum kernel (using a 1-rep ZZFeatureMap as example)
#     feature_dim = X_train.shape[1]
#     feature_map = ZZFeatureMap(feature_dimension=feature_dim, reps=1, entanglement="full")
#     backend = Aer.get_backend("aer_simulator_statevector")

#     q_kernel = FidelityQuantumKernel(feature_map=feature_map)

#     # 3) Initialize M and other parameters
#     n, d = X_train.shape
#     M = np.eye(d, dtype='float32')
#     L = 10.0  # or any bandwidth/scale factor

#     # 4) RFM Iterations
#     for i in range(iters):
#         print(f"[DEBUG] Iteration {i+1}/{iters}")

#         # Compute quantum kernel on (M-transformed) training data
#         K_train = quantum_kernel_matrix(X_train, X_train, q_kernel, M=M, do_encode=True)
#         sol = solve(K_train + reg * np.eye(len(K_train)), y_train.numpy()).T
        
#         # Optional training accuracy
#         if train_acc:
#             preds_train = (sol @ K_train).T
#             y_pred = torch.from_numpy(preds_train)
#             preds_c = torch.argmax(y_pred, dim=-1)
#             labels_c = torch.argmax(y_train, dim=-1)
#             acc_train = (labels_c == preds_c).sum().item() / len(labels_c)
#             print("Round", i, "Train Acc =", acc_train)

#         # Evaluate on test set
#         K_test = quantum_kernel_matrix(X_train, X_test, q_kernel, M=M, do_encode=True)
#         preds = (sol @ K_test).T
#         mse = np.mean((preds - y_test.numpy())**2)
#         print("Round", i, "MSE =", mse)

#         # Classification accuracy if needed
#         if classif:
#             y_pred = torch.from_numpy(preds)
#             preds_c = torch.argmax(y_pred, dim=-1)
#             labels_c = torch.argmax(y_test, dim=-1)
#             acc_test = (labels_c == preds_c).sum().item() / len(labels_c)
#             print("Round", i, "Acc =", acc_test)

#         # 5) Update M using quantum-based gradient
#         M = get_grads(X_train, sol, L, M, q_kernel, batch_size=batch_size, do_encode=True).astype('float32')
        
#         if name is not None:
#             hickle.dump(M, f'saved_Ms/M_{name}_{i}.h')

#     # Final pass
#     K_train = quantum_kernel_matrix(X_train, X_train, q_kernel, M=M, do_encode=True)
#     sol = solve(K_train + reg * np.eye(len(K_train)), y_train.numpy()).T

#     K_test = quantum_kernel_matrix(X_train, X_test, q_kernel, M=M, do_encode=True)
#     preds = (sol @ K_test).T
#     mse = np.mean((preds - y_test.numpy())**2)
#     print("Final MSE:", mse)

#     if classif:
#         y_pred = torch.from_numpy(preds)
#         preds_c = torch.argmax(y_pred, dim=-1)
#         labels_c = torch.argmax(y_test, dim=-1)
#         acc_test = (labels_c == preds_c).sum().item() / len(labels_c)
#         print("Final Acc:", acc_test)

#     return M, mse




import numpy as np
import torch
from numpy.linalg import solve
from tqdm import tqdm
import hickle
import time

import qiskit
from qiskit_aer import Aer
from qiskit.circuit.library import ZZFeatureMap
from qiskit_machine_learning.kernels import FidelityQuantumKernel
from qiskit.quantum_info import state_fidelity

def encode_features(X):
    """
    Scales each column of X to [0, 1], then multiplies by pi.
    X: NumPy array of shape (n_samples, d).
    Returns:
        scaled_X: the same shape, but scaled to [0, pi].
    """
    X = X.copy()
    for j in range(X.shape[1]):
        col_min = X[:, j].min()
        col_max = X[:, j].max()
        if abs(col_max - col_min) < 1e-12:
            X[:, j] = 0.0
        else:
            X[:, j] = (X[:, j] - col_min) / (col_max - col_min)
    X *= np.pi
    return X

def quantum_kernel_matrix(X1, X2, q_kernel, M=None, do_encode=True):
    """
    Compute the quantum kernel matrix for inputs X1 and X2,
    optionally transforming each vector x -> sqrt(M) * x first.

    Args:
        X1, X2: torch.Tensor or np.ndarray of shape (n_samples, n_features)
        q_kernel: an instance of Qiskit's FidelityQuantumKernel (or similar)
        M: np.ndarray (d x d), the RFM matrix to incorporate. If None, no transformation.
        do_encode: whether to scale/encode data [0,1]->[0, pi] before the quantum feature map.

    Returns:
        K: np.ndarray of shape (n_samples_X1, n_samples_X2)
    """
    # Convert to NumPy if needed
    if isinstance(X1, torch.Tensor):
        X1 = X1.cpu().numpy()
    if isinstance(X2, torch.Tensor):
        X2 = X2.cpu().numpy()

    # Optionally encode from [0,1] -> [0, pi]
    if do_encode:
        X1 = encode_features(X1)
        X2 = encode_features(X2)

    # If M is provided, transform data by sqrt(M)
    if M is not None:
        # Compute sqrt(M) via a Cholesky factorization (ensure M is positive-definite)
        sqrtM = np.real_if_close(np.linalg.cholesky(M))
        X1 = X1 @ sqrtM
        X2 = X2 @ sqrtM

    print("[DEBUG] Evaluating Quantum Kernel...")
    start_time = time.time()
    K = q_kernel.evaluate(x_vec=X1, y_vec=X2)
    end_time = time.time()
    print("[DEBUG] Quantum Kernel evaluated successfully! Shape =", K.shape)
    print("[DEBUG] Time = %.2f seconds" % (end_time - start_time))
    return K

def get_grads(X, sol, L, M, q_kernel, batch_size=2, do_encode=True):
    """
    Quantum-based gradient update that uses the quantum kernel (with M-transformation)
    in place of a classical kernel.

    Args:
        X: training data (torch.Tensor or np.ndarray)
        sol: solution vector from the last solve(...) call
        L: scalar parameter (bandwidth/scale)
        M: the current RFM matrix (to be incorporated into the kernel)
        q_kernel: FidelityQuantumKernel instance
        batch_size: mini-batch size for accumulating gradient outer products
        do_encode: whether to apply the encoding (scale data to [0, pi])

    Returns:
        M_new: updated version of M
    """
    print("[DEBUG] Entering get_grads...")
    num_samples = min(len(X), 1000)
    indices = np.random.choice(len(X), size=num_samples, replace=False)
    x = X[indices, :]

    print("[DEBUG] Computing quantum kernel matrix for gradients...")
    # Compute the kernel matrix using the current M (data are transformed via sqrt(M))
    K = quantum_kernel_matrix(X, x, q_kernel, M=M, do_encode=do_encode)
    print("[DEBUG] Quantum kernel matrix computed for gradient shape:", K.shape)

    # Convert solution vector to torch
    a1 = torch.from_numpy(sol.T).float()
    n, d = X.shape
    n, c = a1.shape
    m, d = x.shape

    # First step: compute (sol^T * (X @ M))
    a1 = a1.reshape(n, c, 1)
    # Here we use M to transform X classically
    X1 = torch.from_numpy(X).float() @ torch.from_numpy(M).float()
    X1 = X1.reshape(n, 1, d)
    step1 = a1 @ X1
    del a1, X1
    step1 = step1.reshape(-1, c * d)

    # Second step: use kernel matrix K
    step2 = torch.from_numpy(K).float().T @ step1
    del step1
    step2 = step2.reshape(-1, c, d)

    # Third step: compute other term involving sol and K
    a2 = torch.from_numpy(sol).float()
    step3 = (a2 @ torch.from_numpy(K).float()).T
    del K, a2
    step3 = step3.reshape(m, c, 1)
    x1 = torch.from_numpy(x).float() @ torch.from_numpy(M).float()  # x @ M
    x1 = x1.reshape(m, 1, d)
    step3 = step3 @ x1

    # Compute final gradient G
    G = (step2 - step3) * -1.0 / L

    # Accumulate outer products into updated M
    M_new = 0.0
    batches = torch.split(G, batch_size)
    for i in range(len(batches)):
        grad = batches[i]
        gradT = torch.transpose(grad, 1, 2)
        M_new += torch.sum(gradT @ grad, dim=0).cpu()
        del grad, gradT
    M_new /= len(G)
    M_new = M_new.numpy()
    return M_new

def get_data(loader):
    """
    Gather all data from a PyTorch DataLoader into tensors (X, y).
    For debugging, only the first 500 samples are taken.
    """
    X_list, y_list = [], []
    for idx, batch in enumerate(loader):
        inputs, labels = batch
        X_list.append(inputs)
        y_list.append(labels)

    X = torch.cat(X_list, dim=0)
    y = torch.cat(y_list, dim=0)
    X = X[:500]
    y = y[:500]

    return X, y


#do_encode is always true so, it always encode
#such as 0 iter => encode
# 1 iter => encode(encode)
# 2 iter => encode(encode(encode)))
def q_rfm(train_loader, test_loader, iters=3, name=None, batch_size=2,
          reg=1e-3, train_acc=False, loader=True, classif=True):
    """
    Quantum version of the Recursive Feature Machine (RFM) that incorporates M.
    Here, data are transformed using sqrt(M) before being fed into the quantum kernel.
    """
    print("[DEBUG] Entered q_rfm function...")
    
    # 1) Get raw training and testing data
    if loader:
        X_train, y_train = get_data(train_loader)
        X_test, y_test   = get_data(test_loader)
    else:
        X_train, y_train = train_loader
        X_test, y_test   = test_loader
        X_train = torch.from_numpy(X_train).float()
        y_train = torch.from_numpy(y_train).float()
        X_test  = torch.from_numpy(X_test).float()
        y_test  = torch.from_numpy(y_test).float()

    print("[DEBUG] X_train shape:", X_train.shape, "y_train shape:", y_train.shape)
    print("[DEBUG] X_test shape:", X_test.shape, "y_test shape:", y_test.shape)
    print(X_train.shape[1])
    # 2) Build the quantum kernel using a ZZFeatureMap (1 repetition)
    feature_dim = 8#X_train.shape[1]
    feature_map = ZZFeatureMap(feature_dimension=feature_dim, reps=1, entanglement="full")
    backend = Aer.get_backend("aer_simulator_statevector")
    q_kernel = FidelityQuantumKernel(feature_map=feature_map,enforce_psd=False)

    # 3) Initialize M (start with identity) and set parameter L
    n, d = X_train.shape
    M = np.eye(d, dtype='float32')
    L = 10.0

    # 4) RFM Iterations
    for i in range(iters):
        print(f"[DEBUG] Iteration {i+1}/{iters}")

        # Compute quantum kernel on training data (with data transformed by sqrt(M))
        K_train = quantum_kernel_matrix(X_train, X_train, q_kernel, M=M, do_encode=True)
        sol = solve(K_train + reg * np.eye(len(K_train)), y_train.numpy()).T

        # Optionally compute training accuracy
        if train_acc:
            preds_train = (sol @ K_train).T
            y_pred = torch.from_numpy(preds_train)
            preds_c = torch.argmax(y_pred, dim=-1)
            #labels_c = torch.argmax(y_train, dim=-1)
            acc_test = (y_test == preds_c).sum().item() / y_test.size(0)
            print("Round", i, "Train Acc =", acc_train)

        # Evaluate on test set
        K_test = quantum_kernel_matrix(X_train, X_test, q_kernel, M=M, do_encode=True)
        preds = (sol @ K_test).T
        mse = np.mean((preds - y_test.numpy())**2)
        print("Round", i, "MSE =", mse)

        # Optional classification accuracy on test set
        if classif:
            y_pred = torch.from_numpy(preds)
            preds_c = torch.argmax(y_pred, dim=-1)
            labels_c = torch.argmax(y_test, dim=-1)
            acc_test = (labels_c == preds_c).sum().item() / len(labels_c)
            print("Round", i, "Acc =", acc_test)

        # Update M using the quantum-based gradient update
        M = get_grads(X_train, sol, L, M, q_kernel, batch_size=batch_size, do_encode=True).astype('float32')
        if name is not None:
            hickle.dump(M, f'saved_Ms/M_{name}_{i}.h')

    # Final pass: recompute kernel and evaluate final performance
    K_train = quantum_kernel_matrix(X_train, X_train, q_kernel, M=M, do_encode=True)
    sol = solve(K_train + reg * np.eye(len(K_train)), y_train.numpy()).T
    K_test = quantum_kernel_matrix(X_train, X_test, q_kernel, M=M, do_encode=True)
    preds = (sol @ K_test).T
    mse = np.mean((preds - y_test.numpy())**2)
    print("Final MSE:", mse)
    if classif:
        y_pred = torch.from_numpy(preds)
        preds_c = torch.argmax(y_pred, dim=-1)
        labels_c = torch.argmax(y_test, dim=-1)
        acc_test = (labels_c == preds_c).sum().item() / len(labels_c)
        print("Final Acc:", acc_test)
    return M, mse
