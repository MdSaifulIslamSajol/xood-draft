import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from skimage import filters
from sklearn.model_selection import train_test_split
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.datasets import fashion_mnist, mnist, cifar10, cifar100 as cf100
from scipy.io import loadmat
import torchvision
import torch
import pickle
from pathlib import Path
import os
import OpenOOD
from OpenOOD import openood
# from . import openood
import sys
sys.path.insert(0, '..')  # Enable import from parent folder.

#sys.modules['OpenOOD'] = OpenOOD
#sys.modules['openood'] = openood
#from openood import utils
#sys.modules['OpenOOD.openood.utils'] = utils
#from OpenOOD.openood.utils import config
#from OpenOOD.openood_id_ood_and_model_mnist import id_dataloader_from_openood_repo_mnist, ood_dataloader_from_openood_repo_mnist
from OpenOOD.openood_id_ood_and_model_cifar10 import id_dataloader_from_openood_repo_cifar10

img_shape = (32, 32, 3)
imagenet_transform = torchvision.transforms.Compose([
    torchvision.transforms.Resize(256),
    torchvision.transforms.CenterCrop(224),
    torchvision.transforms.ToTensor()
])

cifar_size_transform = torchvision.transforms.Compose([
    torchvision.transforms.Resize(32),
    torchvision.transforms.CenterCrop(32),
    torchvision.transforms.ToTensor()
])


def load_dataset(name):
    print("data.py ==> load_dataset()")
    global img_shape
    if name == "cifar10":
        img_shape = (32, 32, 3)
        return load_cifar10_id_data_from_openood()
        # return load_data(cifar10.load_data())
        
    elif name == "svhn":
        img_shape = (32, 32, 3)
        return load_data(load_svhn_data())
    
    elif name == "mnist":
        img_shape = (28, 28, 3)
        # return load_data(mnist.load_data())
        print("calling MINSIT")
        return load_mnist_data_from_openood()
    
    elif name == "cifar100":
        img_shape = (32, 32, 3)
        return load_cifar100_id_data_from_openood()
    
    elif name == "imagenet":
        img_shape = (224, 224, 3)
        # return imagenet_validation()
        return load_imagenet_id_data_from_openood()
    
    else:
        raise Exception(f"Unknown dataset: {name}")


def scale_and_save_in_df(images, labels, scale=True):
    print("data.py ==> scale_and_save_in_df()")
    if scale:
        print("scale is getting in")
        images = images / 255
    print("images shape: ", images.shape[0])
    print("type of images data: ", type(images))
    images = images.reshape((images.shape[0], -1))
    print("images after reshaped: ", images.shape)
    df = pd.DataFrame(images, columns=images.shape[1] * ["data"])
    df["label"] = labels
    return df


def load_data(datasets):
    print("data.py ==> load_data()")
    (train_images, train_labels), (test_images, test_labels) = datasets

    test_images, val_images, test_labels, val_labels = train_test_split(test_images, test_labels,
                                                                        stratify=test_labels, test_size=0.5,
                                                                        random_state=42)
    print("train images: ", train_images.shape)
    train = scale_and_save_in_df(train_images, train_labels)
    print("After scale and save train images: ", train.shape)
    val = scale_and_save_in_df(val_images, val_labels)
    test = scale_and_save_in_df(test_images, test_labels)
    train.name, val.name, test.name = "Train", "Val", "Test"
    return {"Train": train, "Val": val, "Test": test}


def get_images_and_labels(df: pd.DataFrame, labels=True, chw=False):
    print("data.py ==> get_images_and_labels()")
    # print("It is failing get_images_and_labels")
    images = df["data"].to_numpy()
    print("Image shape: ", *img_shape)
    print("Image Incoming shape: ", *images.shape)
    images = images.reshape(images.shape[0], *img_shape)
    if chw:
        images = np.moveaxis(images, 3, 1)
    if labels:
        return images, to_categorical(df["label"])
    else:
        return images


def quantize_pixels(data):
    print("data.py ==> quantize_pixels()")
    return np.round(255 * data) / 255


def rotated(df: pd.DataFrame, plot=False):
    print("data.py ==> rotated()")

    df_pos, df_neg = train_test_split(df, test_size=.5)
    pos, neg = df_pos["data"].to_numpy(), df_neg["data"].to_numpy()
    pos, neg = pos.reshape(
        pos.shape[0], *img_shape), neg.reshape(neg.shape[0], *img_shape)
    images = np.concatenate(
        [np.rot90(pos, k=1, axes=(1, 2)), np.rot90(neg, k=-1, axes=(1, 2))], axis=0)
    if plot:
        fig, axs = plt.subplots(3, 3)
        for i, ax in enumerate(np.concatenate(axs)):
            ax.imshow(images[-i][:, :, 0], cmap="Greys")
            ax.set_title(df_neg["label"].iloc[-i])
        plt.savefig("rotated.png")
    images = images.reshape(images.shape[0], -1)
    df = pd.DataFrame(images, columns=images.shape[1] * ["data"])
    df["label"] = np.concatenate(
        [df_pos["label"].to_numpy(), df_neg["label"].to_numpy()])
    return df


# Same as Taylor 2018
def uniform(n, dim):
    print("data.py ==> uniform()")

    df = pd.DataFrame(np.random.uniform(
        0, 1, (n, dim)), columns=dim * ["data"])
    df["label"] = np.NaN
    return df


# Same as Taylor 2018
def gaussian(n, dim):
    print("data.py ==> gaussian()")

    df = pd.DataFrame(np.clip(np.random.normal(loc=.5, scale=1, size=(n, dim)), a_min=0, a_max=1),
                      columns=dim * ["data"])
    df["label"] = np.NaN
    return df


def gaussian_noise(images, var, a_min=0, a_max=1):
    print("data.py ==> gaussian_noise()")

    return np.clip(images + np.random.normal(0, np.sqrt(var), images.shape), a_min=a_min, a_max=a_max)


def awgn(df: pd.DataFrame):
    print("data.py ==> awgn()")

    print(f"Creating Noisy Set", flush=True)
    noisy = df.sample(frac=1).reset_index(drop=True)
    variance = np.linspace(0, 2 ** (1 / 4), 100) ** 4
    # variance = np.logspace(-3, 1, 20)
    noisy["data"] = np.concatenate([gaussian_noise(noisy["data"].loc[ids].to_numpy(), var) for ids, var in
                                    zip(np.array_split(noisy.index.to_numpy(), len(variance)), variance)], axis=0)
    return noisy


def mixed_up(df: pd.DataFrame):
    print("data.py ==> mixed_up()")

    print(f"Creating Mixed Up Set", flush=True)
    df_mixed_up = df[["data", "label"]].sample(frac=1.).reset_index(drop=True)
    #fracs = [.5, .6, .7, .8, .9]
    fracs = np.random.beta(0.4, 0.6, 5)
    df_mixed_up["data"] = np.concatenate(
        [frac * df_mixed_up["data"].loc[ids].to_numpy() + (1 - frac) * df["data"].sample(len(ids)).to_numpy()
         for ids, frac in zip(np.array_split(df_mixed_up.index.to_numpy(), len(fracs)), fracs)],
        axis=0)
    return df_mixed_up


def uniform_mixed_up(df: pd.DataFrame):
    print("data.py ==> uniform_mixed_up()")

    print(f"Creating Uniform Mixed Up Set", flush=True)
    df_mixed_up = df[["data", "label"]].sample(frac=1.).reset_index(drop=True)
    uniform = np.random.uniform(0, 1, (len(df), np.prod(img_shape)))
    fracs = np.linspace(0, 1, 50)
    df_mixed_up["data"] = np.concatenate(
        [frac * df_mixed_up["data"].loc[ids].to_numpy() + (1 - frac) * uniform[ids, :]
         for ids, frac in zip(np.array_split(df_mixed_up.index.to_numpy(), len(fracs)), fracs)],
        axis=0)
    return df_mixed_up


def blurry(df: pd.DataFrame):
    print("data.py ==> blurry()")

    print(f"Creating Blurry Set", flush=True)
    df = df[["data", "label"]].sample(frac=1).reset_index(drop=True)
    images = df["data"].to_numpy().reshape(len(df), *img_shape)
    df["data"] = gaussian_blur(images).reshape(len(df), -1)
    return df


def gaussian_blur(images: np.ndarray, std_range=(.1, 5)):
    print("data.py ==> gaussian_blur()")

    stds = np.linspace(*std_range, 20)
    ids = np.arange(len(images))
    img_list = []
    for ids, std in zip(np.array_split(ids, len(stds)), stds):
        img_list += [filters.gaussian(images[i],
                                      std, multichannel=True) for i in ids]
    return np.concatenate(img_list, axis=0)


def targeted(set_name):
    print("data.py ==> targeted()")

    data_name = set_name + "_adver_targeted"
    data = pd.read_pickle(
        f"results/datasets/adversarial/{set_name}/{data_name}.pkl")
    data = data[["data"]]
    data["label"] = np.NaN
    return data


def non_targeted(set_name):
    print("data.py ==> non_targeted()")

    data_name = set_name + "_adver_non_targeted"
    data = pd.read_pickle(
        f"results/datasets/adversarial/{set_name}/{data_name}.pkl")
    data = data[["data"]]
    data["label"] = np.NaN
    return data


def imagenet():
    data = pd.read_pickle('out_of_distribution/ImageNet.pkl')
    data = data.iloc[:10, :]
    data["label"] = np.NaN
    return data


def isun_scaled():
    data = pd.read_pickle('out_of_distribution/iSUN.pkl')
    data["label"] = np.NaN
    return data


def lsun_scaled():
    data = pd.read_pickle('out_of_distribution/lSUN.pkl')
    data["label"] = np.NaN
    return data


def lsun_resized():
    data = pd.read_pickle('out_of_distribution/lSUN_resize.pkl')
    data["label"] = np.NaN
    return data


def imagenet_resized():
    data = pd.read_pickle('out_of_distribution/imagenet_resize.pkl')
    data["label"] = np.NaN
    return data


def fashion():
    (x_train, y_train), (x_test, y_test) = fashion_mnist.load_data()
    return scale_and_save_in_df(x_test, np.nan)


def cifar100():
    return load_data(cf100.load_data())


def cifar100_as_ood():
    df = cifar100()["Test"]
    #df = df.iloc[:10, :]
    df["label"] = np.nan
    return df


def load_svhn_data(test_only=False):
    test = loadmat("models/svhn/test_32x32.mat")
    x_test = test['X']
    x_test = np.moveaxis(x_test, -1, 0)
    y_test = test['y']
    y_test[y_test == 10] = 0
    if test_only:
        return x_test, y_test
    train = loadmat("models/svhn/train_32x32.mat")
    x_train = train['X']
    x_train = np.moveaxis(x_train, -1, 0)
    y_train = train['y']
    y_train[y_train == 10] = 0

    return (x_train, y_train), (x_test, y_test)


def imagenet_validation(debug=False):
    dataset = torchvision.datasets.ImageFolder(root="imagenet/dummy" if debug else "imagenet/imagenet_val",
                                               transform=imagenet_transform)
    loader = torch.utils.data.DataLoader(
        dataset, shuffle=False, batch_size=len(dataset))
    x = next(iter(loader))[0].numpy()
    x = np.moveaxis(x, 1, 3)
    y = np.array(dataset.targets)
    x_train, x_test, y_train, y_test = train_test_split(
        x, y, stratify=y, test_size=0.5, random_state=42)
    train = scale_and_save_in_df(x_train, y_train, scale=False)
    test = scale_and_save_in_df(x_test, y_test, scale=False)
    return {"Train": train, "Val": train, "Test": test}


def svhn_as_ood():
    return scale_and_save_in_df(load_svhn_data(test_only=True)[0], np.nan)


def cifar10_as_ood():
    df = load_data(cifar10.load_data())["Test"]
    print("Out of distribtuion dataset shape: ", df.shape)
    df = df.iloc[:10, :]
    df["label"] = np.nan
    return df


def rotated_mnist():
    (x_train, y_train), (x_test, y_test) = mnist.load_data()
    df = scale_and_save_in_df(x_test, y_test)
    df = rotated(df[df["label"] != 0])
    df["label"] = np.nan
    return df


def augmented2(df):
    print("data.py ==> augmented2()")

    print(f"Creating Augmented2 Set", flush=True)
    df = df.sample(frac=1.0).reset_index(drop=True)
    generator = ImageDataGenerator(
        rotation_range=30,
        fill_mode="nearest",
        width_shift_range=0.2,
        height_shift_range=0.2,
        vertical_flip=True,
        brightness_range=[.2, 2],
        zoom_range=[.3, 0.9],
    )
    x = get_images_and_labels(df, labels=False)
    generator.fit(x)
    x, y = generator.flow(x, df["label"], batch_size=len(df)).next()
    df = df[["data", "label"]].copy()
    df["data"] = np.reshape(x / 255, (len(df), -1))
    df["label"] = y
    return df


def augmented(df):
    print("data.py ==> augmented()")

    print(f"Creating Augmented Set", flush=True)
    df = df.sample(frac=1.0).reset_index(drop=True)
    generator = ImageDataGenerator(
        rotation_range=90,
        width_shift_range=0.2,
        height_shift_range=0.2,
        horizontal_flip=True,
        brightness_range=[.2, 2],
        zoom_range=[.9, 1.1],
    )
    x = get_images_and_labels(df, labels=False)
    print("It is passing get_images_and_labels")
    generator.fit(x)
    x, y = generator.flow(x, df["label"], batch_size=len(df)).next()
    df = df[["data", "label"]].copy()
    df["data"] = np.reshape(x / 255, (len(df), -1))
    df["label"] = y
    return df


def scale_and_shift(df, scale):
    print("data.py ==> scale_and_shift()")

    x_min = np.linspace(*np.sort([0, 1 - scale]), 20)
    df["data"] = np.clip(np.concatenate([m + scale * df["data"].loc[ids].to_numpy() for ids, m in
                                         zip(np.array_split(df.index.to_numpy(), len(x_min)), x_min)], axis=0),
                         a_min=0., a_max=1.)
    return df


def shifted(df):
    print("data.py ==> shifted()")

    print(f"Creating Shifted Set", flush=True)
    df = df.sample(frac=1).reset_index(drop=True)
    scale = np.logspace(-3, 3, 40, base=2)
    df["data"] = np.concatenate([scale_and_shift(df["data"].loc[ids], s) for ids, s in
                                 zip(np.array_split(df.index.to_numpy(), len(scale)), scale)], axis=0)
    return df


def calibration(df):
    print("data.py ==> calibration()")

    cal_set = df[["data", "label"]].sample(frac=1.).reset_index(drop=True)

    def clean(x):
        print("data.py ==> clean()")

        x = x.reset_index(drop=True)
        return x

    def blurry_to_shifted(x):
        print("data.py ==> blurry_to_shifted()")

        return shifted(blurry(x))

    def awgn_to_shifted(x):
        print("data.py ==> awgn_to_shifted()")

        return shifted(awgn(x))

    mappings = clean, augmented, augmented2, mixed_up, awgn_to_shifted, blurry_to_shifted
    result = {}
    for f in mappings:
        df = f(cal_set)
        df["data"] = quantize_pixels(df["data"])
        result[f.__name__] = df
    return result


def distorted(df):
    print("data.py ==> distorted()")

    df = df[["data", "label"]].sample(frac=1.).reset_index(drop=True)
    datasets = {
        "Clean": df,
        "Uniform MixUp": uniform_mixed_up(df),
        "Shifted": shifted(df),
        "Noisy": awgn(df),
        "Blurry": blurry(df),
        "Shifted -> Blurry": blurry(shifted(df)),
        "Shifted -> Noisy": awgn(shifted(df)),
        "Noisy -> Shifted": shifted(awgn(df)),
        "Augmented": augmented(df),
        "Augmented2": augmented2(df),
        "MixUp": mixed_up(df),
    }
    for dataset in datasets.values():
        dataset["data"] = quantize_pixels(dataset["data"])
    return datasets


def dtd():
    df = pd.read_pickle("out_of_distribution/dtd.pkl")
    df["label"] = np.NaN
    return df


def food():
    df = pd.read_pickle("out_of_distribution/food.pkl")
    df["label"] = np.NaN
    return df


def calt():
    df = pd.read_pickle("out_of_distribution/Calt.pkl")
    df["label"] = np.NaN
    return df


def CUB():
    df = pd.read_pickle("out_of_distribution/CUB.pkl")
    df["label"] = np.NaN
    return df


def places():
    df = pd.read_pickle("out_of_distribution/places.pkl")
    df["label"] = np.NaN
    return df


def stanford_dogs():
    df = pd.read_pickle("out_of_distribution/stanford_dogs.pkl")
    df["label"] = np.NaN
    return df


def out_of_dist(dataset_name, debug=False):
    datasets = {
        # "Uniform": uniform(10000, np.prod(img_shape)),
        #  "Gaussian": gaussian(10000, np.prod(img_shape)),
    }
    if debug:
        return datasets
    if img_shape == (32, 32, 3):
        print("******Image Shape 32-32-3****")
        # datasets.update({
        # "TinyImageNet (Crop)": imagenet(),
        # "TinyImageNet (Resize)": imagenet_resized(),
        # "LSUN (Crop)": lsun_scaled(),
        # "LSUN (Resize)": lsun_resized(),
        # "iSUN": isun_scaled(),
        # "DTD": dtd(),
        # "Stanford Dogs": stanford_dogs(),
        # "food": food(),
        # "CUB": CUB(),
        # "Calt": calt(),
        # "places": places()
        # })
        # for dataset in ['Places' ,'DTD']:
        #     path = f"imagenet/DTD/images" if dataset == "DTD" else f"imagenet/{dataset}"
        #     ood = torchvision.datasets.ImageFolder(root=path,
        #                                             transform = cifar_size_transform)
        #     ood = torch.utils.data.DataLoader(ood, batch_size=len(ood), shuffle=True)
        #     x = next(iter(ood))[0].numpy()
        #     x = np.moveaxis(x, 1, 3)
        #     datasets[dataset] = scale_and_save_in_df(x, np.nan, scale=False)
        #     print("\n")
        #     print(dataset)
        #     print(datasets[dataset])
        #     print("\n")

    elif img_shape == (28, 28, 3):
        # datasets.update({
        #   "Fashion MNIST": fashion(),
        #  "Rotated MNIST": rotated_mnist()
        # })
        if dataset_name == "mnist":
            #mnist_data = out_of_dict_from_openood_for_mnist()
            datasets.update({
                #  "Fashion MNIST": mnist_data['fashionmnist'],
                #  "Not MNIST": mnist_data['notmnist'],
            })

    if dataset_name == "mnist":
        mnist_ood = out_of_dict_from_openood_for_mnist()
        datasets.update({
            "Cifar10": mnist_ood["cifar10"],
            "NonMNIST": mnist_ood["notmnist"],
            "FashionMNIST": mnist_ood["fashionmnist"],
            "Tin": mnist_ood["tin"],
            "Places": mnist_ood["places"],
            "Texture": mnist_ood["texture"],
        })
    elif dataset_name == "cifar10":
        cifar10_ood = out_of_dict_from_openood_for_cifar10()
        datasets.update({
            # "SVHN": svhn_as_ood(),
            # "Cifar100": cifar100_as_ood()
            
            "SVHN": cifar10_ood["svhn"],
            # "Cifar100": cifar10_ood["cifar100"],
            # "Texture": cifar10_ood["texture"],
            # "Places": cifar10_ood["places"],
            # "MNIST": cifar10_ood["mnist"],
            # "Tiny": cifar10_ood["tin"]
            
            # "SVHN": cifar10_ood["svhn"].iloc[:10, :],
            # "Cifar100": cifar10_ood["cifar100"].iloc[:10, :],
            # "Texture": cifar10_ood["texture"].iloc[:10, :],
            # "Places": cifar10_ood["places"].iloc[:10, :],
            # "MNIST": cifar10_ood["mnist"].iloc[:10, :],
            # "Tiny": cifar10_ood["tin"].iloc[:10, :]
  
        })
    elif dataset_name == "svhn":
        datasets.update({
            "Cifar10": cifar10_as_ood(),
            "Cifar100": cifar100_as_ood()
        })
    elif dataset_name == "cifar100":
        cifar100_ood = out_of_dict_from_openood_for_cifar100()
        datasets.update({
            "SVHN": cifar100_ood["svhn"],
            "Cifar10": cifar100_ood["cifar10"],
            "Texture": cifar100_ood["texture"],
            "Places": cifar100_ood["places"],
            "MNIST": cifar100_ood["mnist"],
            "Tiny": cifar100_ood["tin"]
        })
    elif dataset_name == "imagenet":
        imagenet_ood = out_of_dict_from_openood_for_imagenet()
        datasets.update({
            "Species": imagenet_ood["species"],
            "OpenImageo": imagenet_ood["openimageo"],
            "MNIST": imagenet_ood["mnist"],
            "Imageneto": imagenet_ood["imageneto"],
            "iNaturalist": imagenet_ood["inaturalist"],
            "Texture": imagenet_ood["texture"],
        })
        # for dataset in ['DTD']:
        #     path = f"imagenet/DTD/images" if dataset == "DTD" else f"imagenet/{dataset}"
        #     ood = torchvision.datasets.ImageFolder(root=path,
        #                                            transform=imagenet_transform)
        #     ood = torch.utils.data.DataLoader(ood, batch_size=len(ood), shuffle=True)
        #     x = next(iter(ood))[0].numpy()
        #     x = np.moveaxis(x, 1, 3)
        #     datasets[dataset] = scale_and_save_in_df(x, np.nan, scale=False)
        #     print("\n")
        #     print(dataset)
        #     print(datasets[dataset])
        #     print("\n")

    for name in datasets.keys():
        datasets[name]["data"] = quantize_pixels(datasets[name]["data"])
    return datasets


def load_mnist_data_from_openood():
    print(" data.py =>  load_mnist_data_from_openood()")

    sys.path.insert(
        0, '/home/saiful/confidence-magesh_MR/confidence-magesh/OpenOOD/')

    with open('id_train_mnist28_dataloader.pickle', 'rb') as handle:
        train_loader = pickle.load(handle)
    with open('id_val_mnist28_dataloader.pickle', 'rb') as handle:
        val_loader = pickle.load(handle)
    with open('id_test_mnist28_dataloader.pickle', 'rb') as handle:
        test_loader = pickle.load(handle)

    train_features_dict = next(iter(train_loader))
    val_features_dict = next(iter(val_loader))
    test_features_dict = next(iter(test_loader))

    #  dict_keys(['image_name', 'data', 'data_aux', 'label', 'soft_label', 'index', 'pseudo'])
    print("train_features_dict.keys()", train_features_dict.keys())
    print("val_features_dict.keys()", val_features_dict.keys())
    print("test_features_dict.keys()", test_features_dict.keys())

    # torch.Size([128, 3, 32, 32])
    train_features = train_features_dict["data"]
    train_labels = train_features_dict["label"]  # torch.Size([128])

    val_features = val_features_dict["data"]
    val_labels = val_features_dict["label"]

    test_features = test_features_dict["data"]
    test_labels = test_features_dict["label"]

    print("len(train_features):", len(train_features))
    print("len(train_labels):", len(train_labels))

    print("len(val_features):", len(val_features))
    print("len(val_labels):", len(val_labels))

    print("len(test_features):", len(test_features))
    print("len(test_labels):", len(test_labels))

    # converting to numpy
    train_features = train_features.numpy()
    train_features = np.moveaxis(train_features, 1, 3)  # (1000, 224, 224, 3)
    train_labels = np.array(train_labels)

    val_features = val_features.numpy()
    val_features = np.moveaxis(val_features, 1, 3)
    val_labels = np.array(val_labels)

    test_features = test_features.numpy()
    test_features = np.moveaxis(test_features, 1, 3)
    test_labels = np.array(test_labels)

    train = scale_and_save_in_df(train_features, train_labels, scale=False)
    val = scale_and_save_in_df(val_features, val_labels, scale=False)
    test = scale_and_save_in_df(test_features, test_labels, scale=False)
    print("returning Train, Val and Test Set for mnist\n")
    return {"Train": train, "Val": val, "Test": test}


def out_of_dict_from_openood_for_mnist():
    print("data.py => out_of_dict_from_openood_for_mnist")

    old_path = Path.cwd()
    os.chdir("/home/saiful/confidence-magesh_MR/confidence-magesh/OpenOOD")
    temp_path = Path.cwd()
    print(temp_path)
    sys.path.insert(
        0, '/home/saiful/confidence-magesh_MR/confidence-magesh/OpenOOD/')

    with open('cifar10_loader_for_mnist28.pickle', 'rb') as handle:
        ood_cifar10_for_mnist = pickle.load(handle)

    with open('fashionmnist_loader_for_mnist28.pickle', 'rb') as handle:
        ood_fashionmnist_for_mnist=pickle.load(handle)

    with open('notmnist_loader_for_mnist28.pickle', 'rb') as handle:
        ood_notmnist_for_mnist=pickle.load(handle)

    with open('places_loader_for_mnist28.pickle', 'rb') as handle:
        ood_places_for_mnist=pickle.load(handle)

    with open('texture_loader_for_mnist28.pickle', 'rb') as handle:
        ood_texture_for_mnist=pickle.load(handle)

    with open('tin_loader_for_mnist28.pickle', 'rb') as handle:
        ood_tin_for_mnist=pickle.load(handle)

    fashionmnist_loader = ood_fashionmnist_for_mnist
    notmnist_loader = ood_notmnist_for_mnist
    cifar10_loader = ood_cifar10_for_mnist
    tin_loader = ood_tin_for_mnist
    places_loader = ood_places_for_mnist
    texture_loader = ood_texture_for_mnist

    # cifar10
    print("\n# cifar10  from openood ")
    cifar10_features_dict = next(iter(cifar10_loader))
    print("cifar10_features_dict.keys():", cifar10_features_dict.keys())
    # cifar100_imgtnsr.shape = torch.Size([128, 3, 32, 32])
    cifar10_imgtensr = cifar10_features_dict["data"]
    # cifar100_label.shape = torch.Size([128])
    cifar10_label = cifar10_features_dict["label"]
    cifar10_imgnp = cifar10_imgtensr.numpy()  # shape (128, 3, 32, 32)
    # converting to (128, 32, 32, 3)
    cifar10_imgnp = np.moveaxis(cifar10_imgnp, 1, 3)
    df_cifar10 = scale_and_save_in_df(cifar10_imgnp, np.nan, scale=False)

    # tin
    print("\n# tin  from openood ")
    tin_features_dict  = next(iter(tin_loader))
    tin_imgtensr = tin_features_dict["data"]
    tin_label = tin_features_dict["label"]
    tin_imgnp= tin_imgtensr.numpy()
    tin_imgnp = np.moveaxis(tin_imgnp, 1, 3)
    df_tin = scale_and_save_in_df(tin_imgnp, np.nan, scale=False)

    # # notmnist
    print("\n# notmnist  from openood ")
    notmnist_features_dict  = next(iter(notmnist_loader))
    notmnist_imgtensr = notmnist_features_dict["data"]
    notmnist_label = notmnist_features_dict["label"]
    notmnist_imgnp= notmnist_imgtensr.numpy()
    notmnist_imgnp = np.moveaxis(notmnist_imgnp, 1, 3) # mnist_imgnp.shape (128, 32, 32, 3)
    df_notmnist = scale_and_save_in_df(notmnist_imgnp, np.nan, scale=False)

    # fashionmnist
    print("\n# fashionmnist  from openood ")
    fashionmnist_features_dict  = next(iter(fashionmnist_loader))
    fashionmnist_imgtensr = fashionmnist_features_dict["data"]
    fashionmnist_label = fashionmnist_features_dict["label"]
    fashionmnist_imgnp= fashionmnist_imgtensr.numpy()
    fashionmnist_imgnp = np.moveaxis(fashionmnist_imgnp, 1, 3) # mnist_imgnp.shape (128, 32, 32, 3)
    df_fashionmnist = scale_and_save_in_df(fashionmnist_imgnp, np.nan, scale=False)

    # places
    print("\n# places  from openood ")
    places_features_dict  = next(iter(places_loader))
    places_imgtensr = places_features_dict["data"]
    places_label = places_features_dict["label"]
    places_imgnp= places_imgtensr.numpy()
    places_imgnp = np.moveaxis(places_imgnp, 1, 3)
    df_places = scale_and_save_in_df(places_imgnp, np.nan, scale=False)

    # texture
    print("\n# texture  from openood ")
    texture_features_dict  = next(iter(texture_loader))
    texture_imgtensr = texture_features_dict["data"]
    texture_label = texture_features_dict["label"]
    texture_imgnp= texture_imgtensr.numpy()
    texture_imgnp = np.moveaxis(texture_imgnp, 1, 3)
    df_texture = scale_and_save_in_df(texture_imgnp, np.nan, scale=False)

    ood_datasets = {
        "notmnist" : df_notmnist,
        "fashionmnist" : df_fashionmnist,
        "cifar10": df_cifar10,
        "tin" : df_tin ,
        "places" : df_places,
        "texture" : df_texture,

    }
    print("ood_datasets.keys():", ood_datasets.keys())
    os.chdir(old_path)
    return ood_datasets


def load_imagenet_id_data_from_openood():
    print(" data.py =>  load_imagenet_id_data_from_openood()")

    sys.path.insert(
        0, '/home/saiful/confidence-magesh_MR/confidence-magesh/OpenOOD/')

    train_loader, val_loader, test_loader = id_dataloader_from_openood_repo_imagenet()
    # with open('/data/saiful/imagenet_datasets2/train_loader_for_imagenet_4096_images.pickle', 'rb') as handle:
    #     train_loader = pickle.load(handle)
    # with open('val_loader_for_imagenet_320images.pickle', 'rb') as handle:
    #     val_loader = pickle.load(handle)
    # with open('test_loader_for_imagenet_320images.pickle', 'rb') as handle:
    #     test_loader = pickle.load(handle)

    ## imagenet old testloader and val_loader
    
    # with open('val_loader_for_imagenet.pickle', 'rb') as handle:
    #     val_loader = pickle.load(handle)
    # with open('test_loader_for_imagenet.pickle', 'rb') as handle:
    #     test_loader = pickle.load(handle)
        
    # note: putting val_loader to train_loader
    # train_features_dict = next(iter(val_loader))
    # val_features_dict = next(iter(val_loader))
    # test_features_dict = next(iter(test_loader))
    ##

    ## imagenet  new datalaoder
    # with open('/data/saiful/imagenet_datasets2/val_loader_for_imagenet.pickle', 'rb') as handle:
    #     val_features_dict = pickle.load(handle)
        
    # with open('/data/saiful/imagenet_datasets2/test_loader_for_imagenet.pickle', 'rb') as handle:
    #     test_features_dict = pickle.load(handle)
        
    # train_features_dict = val_features_dict
        
    #  dict_keys(['image_name', 'data', 'data_aux', 'label', 'soft_label', 'index', 'pseudo'])
    # print("train_features_dict.keys()", train_features_dict.keys())
    # print("val_features_dict.keys()", val_features_dict.keys())
    # print("test_features_dict.keys()", test_features_dict.keys())
    

    # # torch.Size([128, 3, 32, 32])
    # train_features = train_features_dict["data"]
    # train_labels = train_features_dict["label"]  # torch.Size([128])

    # val_features = val_features_dict["data"]
    # val_labels = val_features_dict["label"]

    # test_features = test_features_dict["data"]
    # test_labels = test_features_dict["label"]
    
    # # taking only 2k images
    # # train_features = train_features[:2000]
    # # train_labels = train_features[:2000]  

    # # val_features = val_features[:2000]
    # # val_labels = val_features[:2000]

    # # test_features = test_features[:2000]
    # # test_labels = test_features[:2000]

    # print("len(train_features):", len(train_features))
    # print("len(train_labels):", len(train_labels))

    # print("len(val_features):", len(val_features))
    # print("len(val_labels):", len(val_labels))

    # print("len(test_features):", len(test_features))
    # print("len(test_labels):", len(test_labels))
    


    # print("\ntrain_features.shape : ", train_features.shape)
    # print("test_features.shape : ", test_features.shape)
    # print("val_features.shape : ", val_features.shape)
    
    # print("This is minimum value of train_features :",torch.min(train_features))
    # print("This is maximum value of train_features :",torch.max(train_features))

    # print("This is minimum value of val_features :",torch.min(val_features))
    # print("This is maximum value of val_features :",torch.max(val_features))

    # print("This is minimum value of test_features :",torch.min(test_features))
    # print("This is maximum value of test_features :",torch.max(test_features))

    # converting to numpy
    # train_features = train_features.numpy()
    # train_features = np.moveaxis(train_features, 1, 3)  # (1000, 224, 224, 3)
    # train_labels = np.array(train_labels)

    # val_features = val_features.numpy()
    # val_features = np.moveaxis(val_features, 1, 3)
    # val_labels = np.array(val_labels)

    # test_features = test_features.numpy()
    # test_features = np.moveaxis(test_features, 1, 3)
    # test_labels = np.array(test_labels)

    # train = scale_and_save_in_df(train_features, train_labels, scale=False)
    # val = scale_and_save_in_df(val_features, val_labels, scale=False)
    # test = scale_and_save_in_df(test_features, test_labels, scale=False)
    print("returning Train, Val and Test Set for imagenet\n")
    return {"Train": test_features, "Val": val_loader, "Test": test_loader}


def out_of_dict_from_openood_for_imagenet():
    print("data.py => out_of_dict_from_openood_for_imagenet()")

    old_path = Path.cwd()
    os.chdir("/home/saiful/confidence-magesh_MR/confidence-magesh")
    temp_path = Path.cwd()
    sys.path.insert(
        0, '/home/saiful/confidence-magesh_MR/confidence-magesh')

    ##
    # with open('imageneto_loader_for_imagenet_320images.pickle', 'rb') as handle:
    #     ood_dict_for_imageneto = pickle.load(handle)

    # with open('mnist_loader_for_imagenet_320images.pickle', 'rb') as handle:
    #     ood_dict_for_mnist = pickle.load(handle)

    # with open('openimageo_loader_for_imagenet_320images.pickle', 'rb') as handle:
    #     ood_dict_for_openimageo = pickle.load(handle)

    # with open('species_loader_for_imagenet_320images.pickle', 'rb') as handle:
    #     ood_dict_for_species = pickle.load(handle)

    # with open('inaturalist_loader_for_imagenet_320images.pickle', 'rb') as handle:
    #     ood_dict_for_inaturalist = pickle.load(handle)

    # with open('texture_loader_for_imagenet_320images.pickle', 'rb') as handle:
    #     ood_dict_for_texture = pickle.load(handle)
    ##
    
    # commenting only for inat
    with open('imageneto_loader_for_imagenet.pickle', 'rb') as handle:
        ood_dict_for_imageneto = pickle.load(handle)

    with open('mnist_loader_for_imagenet.pickle', 'rb') as handle:
        ood_dict_for_mnist = pickle.load(handle)

    with open('openimageo_loader_for_imagenet.pickle', 'rb') as handle:
        ood_dict_for_openimageo = pickle.load(handle)

    with open('species_loader_for_imagenet.pickle', 'rb') as handle:
        ood_dict_for_species = pickle.load(handle)

    with open('inaturalist_loader_for_imagenet.pickle', 'rb') as handle:
        ood_dict_for_inaturalist = pickle.load(handle)

    with open('texture_loader_for_imagenet.pickle', 'rb') as handle:
        ood_dict_for_texture = pickle.load(handle)
    ##

    species_loader = ood_dict_for_species
    openimageo_loader = ood_dict_for_openimageo
    imageneto_loader = ood_dict_for_imageneto
    mnist_loader = ood_dict_for_mnist
    inaturalist_loader = ood_dict_for_inaturalist
    texture_loader = ood_dict_for_texture

    # species
    print("\n# species  from openood ")
    species_features_dict = next(iter(species_loader))
    print("species_features_dict.keys():", species_features_dict.keys())
    # cifar100_imgtnsr.shape = torch.Size([128, 3, 32, 32])
    species_imgtensr = species_features_dict["data"]
    # cifar100_label.shape = torch.Size([128])
    species_label = species_features_dict["label"]
    species_imgnp = species_imgtensr.numpy()  # shape (128, 3, 32, 32)
    # converting to (128, 32, 32, 3)
    species_imgnp = np.moveaxis(species_imgnp, 1, 3)
    df_species = scale_and_save_in_df(species_imgnp, np.nan, scale=False)

    # openimageo
    print("\n# openimageo  from openood ")
    openimageo_features_dict = next(iter(openimageo_loader))
    openimageo_imgtensr = openimageo_features_dict["data"]
    openimageo_label = openimageo_features_dict["label"]
    openimageo_imgnp = openimageo_imgtensr.numpy()
    # mnist_imgnp.shape (128, 32, 32, 3)
    openimageo_imgnp = np.moveaxis(openimageo_imgnp, 1, 3)
    df_openimageo = scale_and_save_in_df(openimageo_imgnp, np.nan, scale=False)

    # imageneto
    print("\n# imageneto  from openood ")
    imageneto_features_dict = next(iter(imageneto_loader))
    imageneto_imgtensr = imageneto_features_dict["data"]
    imageneto_label = imageneto_features_dict["label"]
    imageneto_imgnp = imageneto_imgtensr.numpy()
    imageneto_imgnp = np.moveaxis(imageneto_imgnp, 1, 3)
    df_imageneto = scale_and_save_in_df(imageneto_imgnp, np.nan, scale=False)

    # inaturalist
    print("\n# inaturalist  from openood ")
    inaturalist_features_dict = next(iter(inaturalist_loader))
    print("inaturalist_features_dict.keys():",
          inaturalist_features_dict.keys())
    # cifar100_imgtnsr.shape = torch.Size([128, 3, 32, 32])
    inaturalist_imgtensr = inaturalist_features_dict["data"]
    # cifar100_label.shape = torch.Size([128])
    inaturalist_label = inaturalist_features_dict["label"]
    inaturalist_imgnp = inaturalist_imgtensr.numpy()  # shape (128, 3, 32, 32)
    # converting to (128, 32, 32, 3)
    inaturalist_imgnp = np.moveaxis(inaturalist_imgnp, 1, 3)
    df_inaturalist = scale_and_save_in_df(
        inaturalist_imgnp, np.nan, scale=False)

    # mnist
    print("\n# mnist  from openood ")
    mnist_features_dict = next(iter(mnist_loader))
    mnist_imgtensr = mnist_features_dict["data"]
    mnist_label = mnist_features_dict["label"]
    mnist_imgnp = mnist_imgtensr.numpy()
    # mnist_imgnp.shape (128, 32, 32, 3)
    mnist_imgnp = np.moveaxis(mnist_imgnp, 1, 3)
    df_mnist = scale_and_save_in_df(mnist_imgnp, np.nan, scale=False)

    # texture
    print("\n# texture  from openood ")
    texture_features_dict = next(iter(texture_loader))
    texture_imgtensr = texture_features_dict["data"]
    texture_label = texture_features_dict["label"]
    texture_imgnp = texture_imgtensr.numpy()
    texture_imgnp = np.moveaxis(texture_imgnp, 1, 3)
    df_texture = scale_and_save_in_df(texture_imgnp, np.nan, scale=False)
    
    print("This is minimum value of species_imgtensr :",torch.min(species_imgtensr))
    print("This is maximum value of species_imgtensr :",torch.max(species_imgtensr))

    print("This is minimum value of openimageo_imgtensr :",torch.min(openimageo_imgtensr))
    print("This is maximum value of openimageo_imgtensr :",torch.max(openimageo_imgtensr))

    print("This is minimum value of imageneto_imgtensr :",torch.min(imageneto_imgtensr))
    print("This is maximum value of imageneto_imgtensr :",torch.max(imageneto_imgtensr))

    print("This is minimum value of inaturalist_imgtensr :",torch.min(inaturalist_imgtensr))
    print("This is maximum value of inaturalist_imgtensr :",torch.max(inaturalist_imgtensr))

    print("This is minimum value of mnist_imgtensr :",torch.min(mnist_imgtensr))
    print("This is maximum value of mnist_imgtensr :",torch.max(mnist_imgtensr))

    print("This is minimum value of texture_imgtensr :",torch.min(texture_imgtensr))
    print("This is maximum value of texture_imgtensr :",torch.max(texture_imgtensr))

    ood_datasets = {
        "species": df_species,
        "openimageo": df_openimageo,
        "imageneto": df_imageneto,
        "mnist": df_mnist,
        "inaturalist": df_inaturalist,
        "texture": df_texture
    }
    print("ood_datasets.keys():", ood_datasets.keys())
    os.chdir(old_path)
    return ood_datasets

def load_cifar10_id_data_from_openood():
    print(" data.py =>  load_cifar10_id_data_from_openood()")

    sys.path.insert(
        0, '/home/saiful/confidence-magesh_MR/confidence-magesh/OpenOOD/')
    train_loader,val_loader, test_loader = id_dataloader_from_openood_repo_cifar10()
    
    # with open('train_loader_for_cifar10.pickle', 'rb') as handle:
    #     train_loader = pickle.load(handle)
    # with open('val_loader_for_cifar10.pickle', 'rb') as handle:
    #     val_loader = pickle.load(handle)
    # with open('test_loader_for_cifar10.pickle', 'rb') as handle:
    #     test_loader = pickle.load(handle)

    # train_features_dict = next(iter(train_loader))
    # val_features_dict = next(iter(val_loader))
    # test_features_dict = next(iter(test_loader))
    
    #  dict_keys(['image_name', 'data', 'data_aux', 'label', 'soft_label', 'index', 'pseudo'])
    # print("train_features_dict.keys()", train_features_dict.keys())
    # print("val_features_dict.keys()", val_features_dict.keys())
    # print("test_features_dict.keys()", test_features_dict.keys())

    # torch.Size([128, 3, 32, 32])
    # train_features = train_features_dict["data"]
    # train_labels = train_features_dict["label"]  # torch.Size([128])

    # val_features = val_features_dict["data"]
    # val_labels = val_features_dict["label"]

    # test_features = test_features_dict["data"]
    # test_labels = test_features_dict["label"]

    # print("len(train_features):", len(train_features))
    # print("len(train_labels):", len(train_labels))

    # print("len(val_features):", len(val_features))
    # print("len(val_labels):", len(val_labels))

    # print("len(test_features):", len(test_features))
    # print("len(test_labels):", len(test_labels))

    # print("This is minimum value of train_features :",torch.min(train_features))
    # print("This is maximum value of train_features :",torch.max(train_features))

    # print("This is minimum value of val_features :",torch.min(val_features))
    # print("This is maximum value of val_features :",torch.max(val_features))

    # print("This is minimum value of test_features :",torch.min(test_features))
    # print("This is maximum value of test_features :",torch.max(test_features))
        
    # # converting to numpy
    # train_features = train_features.numpy()
    # train_features = np.moveaxis(train_features, 1, 3)  # (1000, 224, 224, 3)
    # train_labels = np.array(train_labels)

    # val_features = val_features.numpy()
    # val_features = np.moveaxis(val_features, 1, 3)
    # val_labels = np.array(val_labels)

    # test_features = test_features.numpy()
    # test_features = np.moveaxis(test_features, 1, 3)
    # test_labels = np.array(test_labels)

    # train = scale_and_save_in_df(train_features, train_labels, scale=False)
    # val = scale_and_save_in_df(val_features, val_labels, scale=False)
    # test = scale_and_save_in_df(test_features, test_labels, scale=False)
    print("returning Train, Val and Test Set for CIFAR10 \n")
    return {"Train": train_loader, "Val": val_loader, "Test": test_loader}

def load_cifar10_id_data_from_openood2():
    print(" data.py =>  load_cifar10_id_data_from_openood()")

    sys.path.insert(
        0, '/home/saiful/confidence-magesh_MR/confidence-magesh/OpenOOD/')

    with open('train_loader_for_cifar10.pickle', 'rb') as handle:
        train_loader = pickle.load(handle)
    with open('val_loader_for_cifar10.pickle', 'rb') as handle:
        val_loader = pickle.load(handle)
    with open('test_loader_for_cifar10.pickle', 'rb') as handle:
        test_loader = pickle.load(handle)

    train_features_dict = next(iter(train_loader))
    val_features_dict = next(iter(val_loader))
    test_features_dict = next(iter(test_loader))

    #  dict_keys(['image_name', 'data', 'data_aux', 'label', 'soft_label', 'index', 'pseudo'])
    print("train_features_dict.keys()", train_features_dict.keys())
    print("val_features_dict.keys()", val_features_dict.keys())
    print("test_features_dict.keys()", test_features_dict.keys())

    # torch.Size([128, 3, 32, 32])
    train_features = train_features_dict["data"]
    train_labels = train_features_dict["label"]  # torch.Size([128])

    val_features = val_features_dict["data"]
    val_labels = val_features_dict["label"]

    test_features = test_features_dict["data"]
    test_labels = test_features_dict["label"]

    print("len(train_features):", len(train_features))
    print("len(train_labels):", len(train_labels))

    print("len(val_features):", len(val_features))
    print("len(val_labels):", len(val_labels))

    print("len(test_features):", len(test_features))
    print("len(test_labels):", len(test_labels))

    print("This is minimum value of train_features :",torch.min(train_features))
    print("This is maximum value of train_features :",torch.max(train_features))

    print("This is minimum value of val_features :",torch.min(val_features))
    print("This is maximum value of val_features :",torch.max(val_features))

    print("This is minimum value of test_features :",torch.min(test_features))
    print("This is maximum value of test_features :",torch.max(test_features))
    
    # converting to numpy
    train_features = train_features.numpy()
    train_features = np.moveaxis(train_features, 1, 3)  # (1000, 224, 224, 3)
    train_labels = np.array(train_labels)

    val_features = val_features.numpy()
    val_features = np.moveaxis(val_features, 1, 3)
    val_labels = np.array(val_labels)

    test_features = test_features.numpy()
    test_features = np.moveaxis(test_features, 1, 3)
    test_labels = np.array(test_labels)

    train = scale_and_save_in_df(train_features, train_labels, scale=False)
    val = scale_and_save_in_df(val_features, val_labels, scale=False)
    test = scale_and_save_in_df(test_features, test_labels, scale=False)
    print("returning Train, Val and Test Set for CIFAR10 \n")
    return {"Train": train, "Val": val, "Test": test}


def out_of_dict_from_openood_for_cifar10():
    print("data.py => out_of_dict_from_openood_for_cifar10")

    old_path = Path.cwd()
    os.chdir("/home/saiful/confidence-magesh_MR/confidence-magesh/OpenOOD")
    temp_path = Path.cwd()
    print(temp_path)
    sys.path.insert(
        0, '/home/saiful/confidence-magesh_MR/confidence-magesh/OpenOOD/')
    sys.path.insert(0, '..')
    # loading ood data for cifar10 from openood
    # change directory to /home/saiful/OpenOOD_framework/OpenOOD
    # with open('/home/saiful/OpenOOD_framework/OpenOOD/ood_dataloader_for_cifar10_from_openood_bs128.pickle', 'rb') as handle:
    #     ood_dict_for_cifar = pickle.load(handle)
    # /home/saiful/OpenOOD_framework/OpenOOD/ood_dataloader_for_cifar10_from_openood_bs128.pickle

    ##
    # ood_dict_for_mnist=ood_dataloader_from_openood_repo_mnist()

    # with open('tin_loader_for_cifar10_id.pickle', 'rb') as handle:
    #     ood_tin_for_cifar10 = pickle.load(handle)

    # with open('places_loader_for_cifar10_id.pickle', 'rb') as handle:
    #     ood_places_for_cifar10 = pickle.load(handle)

    # with open('mnist_loader_for_cifar10_id.pickle', 'rb') as handle:
    #     ood_mnist_for_cifar10 = pickle.load(handle)

    with open('svhn_loader_for_cifar10_id.pickle', 'rb') as handle:
        ood_svhn_for_cifar10 = pickle.load(handle)

    # with open('cifar100_loader_for_cifar10_id.pickle', 'rb') as handle:
    #     ood_cifar100_for_cifar10 = pickle.load(handle)

    # with open('texture_loader_for_cifar10_id.pickle', 'rb') as handle:
    #     ood_texture_for_cifar10 = pickle.load(handle)

    ##
    # print("ood_dict_for_cifar.keys():", ood_dict_for_mnist.keys()) #dict_keys(['val', 'nearood', 'farood'])
    # print("ood_dict_for_mnist[nearood].keys():",ood_dict_for_mnist["nearood"].keys()) # dict_keys(['cifar100', 'tin'])
    # print("ood_dict_for_mnist[farood].keys():",ood_dict_for_mnist["farood"].keys()) # dict_keys(['mnist', 'svhn', 'texture', 'place365'])

    # access each dataloader

    # fashionmnist_loader = ood_dict_for_mnist['nearood']['fashionmnist']
    # notmnist_loader = ood_dict_for_mnist['nearood']['notmnist']
    # cifar10_loader = ood_dict_for_mnist['farood']['cifar10']
    # tin_loader = ood_dict_for_mnist['farood']['tin']
    # places_loader = ood_dict_for_mnist['farood']['places365']
    
    svhn_loader = ood_svhn_for_cifar10
    # mnist_loader = ood_mnist_for_cifar10
    # cifar100_loader = ood_cifar100_for_cifar10
    # tin_loader = ood_tin_for_cifar10
    # places_loader = ood_places_for_cifar10
    # texture_loader = ood_texture_for_cifar10

    # # cifar10
    # print("\n# cifar10  from openood ")
    # cifar100_features_dict = next(iter(cifar100_loader))
    # print("cifar10_features_dict.keys():", cifar100_features_dict.keys())
    # # >>> cifar100_features_dict.keys(): dict_keys(['image_name', 'data', 'data_aux', 'label', 'soft_label', 'index', 'pseudo'])
    # # cifar100_imgtnsr.shape = torch.Size([128, 3, 32, 32])
    # cifar100_imgtensr = cifar100_features_dict["data"]
    # # cifar100_label.shape = torch.Size([128])
    # cifar100_label = cifar100_features_dict["label"]
    # cifar100_imgnp = cifar100_imgtensr.numpy()  # shape (128, 3, 32, 32)
    # # converting to (128, 32, 32, 3)
    # cifar100_imgnp = np.moveaxis(cifar100_imgnp, 1, 3)
    # df_cifar100 = scale_and_save_in_df(cifar100_imgnp, np.nan, scale=False)

    # # tin
    # print("\n# tin  from openood ")
    # tin_features_dict = next(iter(tin_loader))
    # tin_imgtensr = tin_features_dict["data"]
    # tin_label = tin_features_dict["label"]
    # tin_imgnp = tin_imgtensr.numpy()
    # tin_imgnp = np.moveaxis(tin_imgnp, 1, 3)
    # df_tin = scale_and_save_in_df(tin_imgnp, np.nan, scale=False)

    # # notmnist
    # print("\n# notmnist  from openood ")
    # mnist_features_dict = next(iter(mnist_loader))
    # mnist_imgtensr = mnist_features_dict["data"]
    # mnist_label = mnist_features_dict["label"]
    # # mnist_imgtensr = mnist_imgtensr[:2000]
    # # mnist_label= mnist_label[:2000]
    # mnist_imgnp = mnist_imgtensr.numpy()
    # # mnist_imgnp.shape (128, 32, 32, 3)
    # mnist_imgnp = np.moveaxis(mnist_imgnp, 1, 3)
    # df_mnist = scale_and_save_in_df(mnist_imgnp, np.nan, scale=False)

    # svhn
    print("\n# svhn  from openood ")
    svhn_features_dict = next(iter(svhn_loader))
    svhn_imgtensr = svhn_features_dict["data"]
    svhn_label = svhn_features_dict["label"]
    # mnist_imgtensr = mnist_imgtensr[:2000]
    # mnist_label= mnist_label[:2000]
    svhn_imgnp = svhn_imgtensr.numpy()
    # mnist_imgnp.shape (128, 32, 32, 3)
    svhn_imgnp = np.moveaxis(svhn_imgnp, 1, 3)
    df_svhn = scale_and_save_in_df(svhn_imgnp, np.nan, scale=False)

    # # places
    # print("\n# places  from openood ")
    # places_features_dict = next(iter(places_loader))
    # places_imgtensr = places_features_dict["data"]
    # places_label = places_features_dict["label"]
    # places_imgnp = places_imgtensr.numpy()
    # places_imgnp = np.moveaxis(places_imgnp, 1, 3)
    # df_places = scale_and_save_in_df(places_imgnp, np.nan, scale=False)

    # # texture
    # print("\n# texture  from openood ")
    # texture_features_dict = next(iter(texture_loader))
    # texture_imgtensr = texture_features_dict["data"]
    # texture_label = texture_features_dict["label"]
    # texture_imgnp = texture_imgtensr.numpy()
    # texture_imgnp = np.moveaxis(texture_imgnp, 1, 3)
    # df_texture = scale_and_save_in_df(texture_imgnp, np.nan, scale=False)
    
    # print("This is minimum value of cifar100_imgtensr :",torch.min(cifar100_imgtensr))
    # print("This is maximum value of cifar100_imgtensr :",torch.max(cifar100_imgtensr))

    # print("This is minimum value of tin_imgtensr :",torch.min(tin_imgtensr))
    # print("This is maximum value of tin_imgtensr :",torch.max(tin_imgtensr))

    # print("This is minimum value of mnist_imgtensr :",torch.min(mnist_imgtensr))
    # print("This is maximum value of mnist_imgtensr :",torch.max(mnist_imgtensr))

    # print("This is minimum value of svhn_imgtensr :",torch.min(svhn_imgtensr))
    # print("This is maximum value of svhn_imgtensr :",torch.max(svhn_imgtensr))

    # print("This is minimum value of places_imgtensr :",torch.min(places_imgtensr))
    # print("This is maximum value of places_imgtensr :",torch.max(places_imgtensr))

    # print("This is minimum value of texture_imgtensr :",torch.min(texture_imgtensr))
    # print("This is maximum value of texture_imgtensr :",torch.max(texture_imgtensr))

    ood_datasets = {
        # "mnist": df_mnist,
        "svhn": df_svhn,
        # "cifar100": df_cifar100,
        # "tin": df_tin,
        # "places": df_places,
        # "texture": df_texture,

    }
    print("ood_datasets.keys():", ood_datasets.keys())
    os.chdir(old_path)
    return ood_datasets


def load_cifar100_id_data_from_openood():
    print(" data.py =>  load_cifar100_id_data_from_openood()")
    from os import chdir
    old_path = Path.cwd()
    print("old_path", old_path)
    # change directory temporarily to OpenOOD
    chdir("/home/saiful/confidence-magesh_MR/confidence-magesh/OpenOOD/")
    temp_path = Path.cwd()
    sys.path.insert(
        0, '/home/saiful/confidence-magesh_MR/confidence-magesh/OpenOOD/')

    with open('train_loader_for_cifar100_n.pickle', 'rb') as handle:
        train_loader = pickle.load(handle)

    with open('val_loader_for_cifar100_n.pickle', 'rb') as handle:
        val_loader = pickle.load(handle)

    with open('test_loader_for_cifar100_n.pickle', 'rb') as handle:
        test_loader = pickle.load(handle)

    #train_loader,val_loader, test_loader=  id_dataloader_from_openood_repo_imagenet()

    train_features_dict = next(iter(train_loader))
    val_features_dict = next(iter(val_loader))
    test_features_dict = next(iter(test_loader))

    #  dict_keys(['image_name', 'data', 'data_aux', 'label', 'soft_label', 'index', 'pseudo'])
    print("train_features_dict.keys()", train_features_dict.keys())
    print("val_features_dict.keys()", val_features_dict.keys())
    print("test_features_dict.keys()", test_features_dict.keys())

    # torch.Size([128, 3, 32, 32])
    train_features = train_features_dict["data"]
    train_labels = train_features_dict["label"]  # torch.Size([128])

    val_features = val_features_dict["data"]
    val_labels = val_features_dict["label"]

    test_features = test_features_dict["data"]
    test_labels = test_features_dict["label"]

    print("len(train_features):", len(train_features))
    print("len(train_labels):", len(train_labels))

    print("len(val_features):", len(val_features))
    print("len(val_labels):", len(val_labels))

    print("len(test_features):", len(test_features))
    print("len(test_labels):", len(test_labels))
    
    print("This is minimum value of train_features :",torch.min(train_features))
    print("This is maximum value of train_features :",torch.max(train_features))

    print("This is minimum value of val_features :",torch.min(val_features))
    print("This is maximum value of val_features :",torch.max(val_features))

    print("This is minimum value of test_features :",torch.min(test_features))
    print("This is maximum value of test_features :",torch.max(test_features))

    # converting to numpy
    train_features = train_features.numpy()
    train_features = np.moveaxis(train_features, 1, 3)  # (1000, 224, 224, 3)
    train_labels = np.array(train_labels)

    val_features = val_features.numpy()
    val_features = np.moveaxis(val_features, 1, 3)
    val_labels = np.array(val_labels)

    test_features = test_features.numpy()
    test_features = np.moveaxis(test_features, 1, 3)
    test_labels = np.array(test_labels)

    train = scale_and_save_in_df(train_features, train_labels, scale=False)
    val = scale_and_save_in_df(val_features, val_labels, scale=False)
    test = scale_and_save_in_df(test_features, test_labels, scale=False)
    print("returning Train, Val and Test Set for cifar10\n")
    chdir(old_path)
    return {"Train": train, "Val": val, "Test": test}


def out_of_dict_from_openood_for_cifar100():
    print("data.py => out_of_dict_from_openood_for_cifar100()")

    old_path = Path.cwd()
    os.chdir("/home/saiful/confidence-magesh_MR/confidence-magesh/OpenOOD")
    temp_path = Path.cwd()
    # loading ood data for cifar10 from openood
    # change directory to /home/saiful/OpenOOD_framework/OpenOOD
    # with open('/home/saiful/OpenOOD_framework/OpenOOD/ood_dataloader_for_cifar10_from_openood_bs128.pickle', 'rb') as handle:
    #     ood_dict_for_cifar = pickle.load(handle)
    # /home/saiful/OpenOOD_framework/OpenOOD/ood_dataloader_for_cifar10_from_openood_bs128.pickle

    ##

    sys.path.insert(
        0, '/home/saiful/confidence-magesh_MR/confidence-magesh/OpenOOD/')

    with open('cifar10_loader_for_cifar100_id_n.pickle', 'rb') as handle:
        ood_dict_cifar10 = pickle.load(handle)

    with open('mnist_loader_for_cifar100_id_n.pickle', 'rb') as handle:
        ood_dict_mnist = pickle.load(handle)

    with open('places_loader_for_cifar100_id_n.pickle', 'rb') as handle:
        ood_dict_places = pickle.load(handle)

    with open('svhn_loader_for_cifar100_id_n.pickle', 'rb') as handle:
        ood_dict_svhn = pickle.load(handle)

    with open('texture_loader_for_cifar100_id_n.pickle', 'rb') as handle:
        ood_dict_texture = pickle.load(handle)

    with open('tin_loader_for_cifar100_id_n.pickle', 'rb') as handle:
        ood_dict_for_tin = pickle.load(handle)

    cifar10_loader = ood_dict_cifar10
    mnist_loader = ood_dict_mnist
    places_loader = ood_dict_places
    svhn_loader = ood_dict_svhn
    texture_loader = ood_dict_texture
    tin_loader = ood_dict_for_tin

    # species
    print("\n# species  from openood ")
    cifar10_features_dict = next(iter(cifar10_loader))
    print("cifar100_features_dict.keys():", cifar10_features_dict.keys())
    # cifar100_imgtnsr.shape = torch.Size([128, 3, 32, 32])
    cifar10_imgtensr = cifar10_features_dict["data"]
    # cifar100_label.shape = torch.Size([128])
    cifar10_label = cifar10_features_dict["label"]
    cifar10_imgnp = cifar10_imgtensr.numpy()  # shape (128, 3, 32, 32)
    # converting to (128, 32, 32, 3)
    cifar10_imgnp = np.moveaxis(cifar10_imgnp, 1, 3)
    df_cifar10 = scale_and_save_in_df(cifar10_imgnp, np.nan, scale=False)

    # imageneto
    print("\n# imageneto  from openood ")
    places_features_dict = next(iter(places_loader))
    places_imgtensr = places_features_dict["data"]
    places_label = places_features_dict["label"]
    places_imgnp = places_imgtensr.numpy()
    places_imgnp = np.moveaxis(places_imgnp, 1, 3)
    df_places = scale_and_save_in_df(places_imgnp, np.nan, scale=False)

    # inaturalist
    print("\n# inaturalist  from openood ")
    svhn_features_dict = next(iter(svhn_loader))
    # print("inaturalist_features_dict.keys():",inaturalist_features_dict.keys())
    # >>> cifar100_features_dict.keys(): dict_keys(['image_name', 'data', 'data_aux', 'label', 'soft_label', 'index', 'pseudo'])
    # cifar100_imgtnsr.shape = torch.Size([128, 3, 32, 32])
    svhn_imgtensr = svhn_features_dict["data"]
    # cifar100_label.shape = torch.Size([128])
    svhn_label = svhn_features_dict["label"]
    svhn_imgnp = svhn_imgtensr.numpy()  # shape (128, 3, 32, 32)
    # converting to (128, 32, 32, 3)
    svhn_imgnp = np.moveaxis(svhn_imgnp, 1, 3)
    df_svhn = scale_and_save_in_df(svhn_imgnp, np.nan, scale=False)

    # mnist
    print("\n# mnist  from openood ")
    mnist_features_dict = next(iter(mnist_loader))
    mnist_imgtensr = mnist_features_dict["data"]
    mnist_label = mnist_features_dict["label"]
    # mnist_imgtensr = mnist_imgtensr[:2000]
    # mnist_label= mnist_label[:2000]
    mnist_imgnp = mnist_imgtensr.numpy()
    # mnist_imgnp.shape (128, 32, 32, 3)
    mnist_imgnp = np.moveaxis(mnist_imgnp, 1, 3)
    df_mnist = scale_and_save_in_df(mnist_imgnp, np.nan, scale=False)

    # texture
    print("\n# texture  from openood ")
    texture_features_dict = next(iter(texture_loader))
    texture_imgtensr = texture_features_dict["data"]
    texture_label = texture_features_dict["label"]
    texture_imgnp = texture_imgtensr.numpy()
    texture_imgnp = np.moveaxis(texture_imgnp, 1, 3)
    df_texture = scale_and_save_in_df(texture_imgnp, np.nan, scale=False)

    # Tin
    print("\n# texture  from openood ")
    tin_features_dict = next(iter(tin_loader))
    tin_imgtensr = tin_features_dict["data"]
    tin_label = tin_features_dict["label"]
    tin_imgnp = tin_imgtensr.numpy()
    tin_imgnp = np.moveaxis(tin_imgnp, 1, 3)
    df_tin = scale_and_save_in_df(tin_imgnp, np.nan, scale=False)

    ood_datasets = {
        "cifar10": df_cifar10,
        "mnist": df_mnist,
        "places": df_places,
        "texture": df_texture,
        "tin": df_tin,
        "svhn": df_svhn
    }
    print("ood_datasets.keys():", ood_datasets.keys())
    
    print("This is minimum value of cifar10_imgtensr :",torch.min(cifar10_imgtensr))
    print("This is maximum value of cifar10_imgtensr :",torch.max(cifar10_imgtensr))

    print("This is minimum value of tin_imgtensr :",torch.min(tin_imgtensr))
    print("This is maximum value of tin_imgtensr :",torch.max(tin_imgtensr))

    print("This is minimum value of mnist_imgtensr :",torch.min(mnist_imgtensr))
    print("This is maximum value of mnist_imgtensr :",torch.max(mnist_imgtensr))

    print("This is minimum value of svhn_imgtensr :",torch.min(svhn_imgtensr))
    print("This is maximum value of svhn_imgtensr :",torch.max(svhn_imgtensr))

    print("This is minimum value of places_imgtensr :",torch.min(places_imgtensr))
    print("This is maximum value of places_imgtensr :",torch.max(places_imgtensr))

    print("This is minimum value of texture_imgtensr :",torch.min(texture_imgtensr))
    print("This is maximum value of texture_imgtensr :",torch.max(texture_imgtensr))
    os.chdir(old_path)
    return ood_datasets


if __name__ == "__main__":
    out_of_dict_from_openood_for_imagenet()
    # out_of_dict_from_openood_for_mnist()
    # print(imagenet_validation())
    # exit()
    # datasets = distorted(svhn_as_ood())
    # datasets.update(out_of_dist("svhn"))
    # for name, df in datasets.items():
    #     print(name)
    #     print("Max:", np.max(df["data"].to_numpy()))
    #     print("Min:", np.min(df["data"].to_numpy()))
