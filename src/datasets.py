import os
import numpy as np

def download_and_cache_dataset(data_dir, filename):
    path = os.path.join(data_dir, filename)
    os.makedirs(data_dir, exist_ok=True)
    
    print(f"\nDataset {filename} not found at '{path}'. Downloading from public source...")
    
    from sklearn.datasets import fetch_openml
    
    if filename == 'mnist.npz':
        data = fetch_openml('mnist_784', version=1, parser='auto', as_frame=False)
        X = data.data.astype(np.float64)
        y = data.target.astype(np.int64)
    elif filename == 'fashion_mnist.npz':
        data = fetch_openml('Fashion-MNIST', version=1, parser='auto', as_frame=False)
        X = data.data.astype(np.float64)
        y = data.target.astype(np.int64)
    elif filename == 'kuzushiji.npz':
        data = fetch_openml('Kuzushiji-MNIST', version=1, parser='auto', as_frame=False)
        X = data.data.astype(np.float64)
        y = data.target.astype(np.int64)
    elif filename == 'usps.npz':
        data = fetch_openml('usps', version=1, parser='auto', as_frame=False)
        X = data.data.astype(np.float64)
        y = data.target.astype(np.int64)
    elif filename == 'breast_cancer.npz':
        from sklearn.datasets import load_breast_cancer
        data = load_breast_cancer()
        X = data.data.astype(np.float64)
        y = data.target.astype(np.int64)
    else:
        raise ValueError(f"Unknown dataset filename: {filename}")
        
    np.savez(path, X=X, y=y)
    print(f"Dataset {filename} successfully downloaded and cached at '{path}'.")

def load_npz_dataset(data_dir, filename, normalize=True):
    path = os.path.join(data_dir, filename)
    if not os.path.exists(path):
        download_and_cache_dataset(data_dir, filename)
        
    data = np.load(path)
    X = data['X']
    y = data['y']
    if normalize and filename in ['mnist.npz', 'fashion_mnist.npz', 'kuzushiji.npz']:
        X = X / 255.0
    return X, y

def load_mnist(data_dir='data'):
    return load_npz_dataset(data_dir, 'mnist.npz', normalize=True)

def load_fashion_mnist(data_dir='data'):
    return load_npz_dataset(data_dir, 'fashion_mnist.npz', normalize=True)

def load_kuzushiji(data_dir='data'):
    return load_npz_dataset(data_dir, 'kuzushiji.npz', normalize=True)

def load_usps(data_dir='data'):
    return load_npz_dataset(data_dir, 'usps.npz', normalize=False)

def load_breast_cancer(data_dir='data'):
    return load_npz_dataset(data_dir, 'breast_cancer.npz', normalize=False)

def load_cifar10():
    try:
        from tensorflow.keras.datasets import cifar10
        (X_cifar, y_cifar), (_, _) = cifar10.load_data()
        X_cifar = X_cifar.reshape(-1, 3072).astype(np.float64) / 255.0
        y_cifar = y_cifar.flatten()
        return X_cifar, y_cifar
    except ImportError:
        print("Tensorflow not installed. Cannot load CIFAR-10.")
        return None, None

def load_all_datasets(data_dir='data'):
    datasets = {}
    print("Loading MNIST...")
    datasets['MNIST'] = load_mnist(data_dir)
    print("Loading Fashion-MNIST...")
    datasets['Fashion'] = load_fashion_mnist(data_dir)
    print("Loading Kuzushiji-MNIST...")
    datasets['Kuzushiji'] = load_kuzushiji(data_dir)
    print("Loading USPS...")
    datasets['USPS'] = load_usps(data_dir)
    print("Loading Breast Cancer...")
    datasets['BreastCancer'] = load_breast_cancer(data_dir)
    return datasets
