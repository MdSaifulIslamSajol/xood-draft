import os
os.environ["CUDA_VISIBLE_DEVICES"]="2"
import torch.multiprocessing
torch.multiprocessing.set_sharing_strategy('file_system')
from pathlib import Path
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from sklearn.metrics import roc_auc_score, det_curve, average_precision_score, roc_curve
from tensorflow.keras.datasets import cifar10, mnist
from sklearn import preprocessing

from confidenciator import Confidenciator, split_features
from data import distorted, calibration, out_of_dist, load_data, load_svhn_data, imagenet_validation
import data
from utils import binary_class_hist, df_to_pdf
from models.load import load_model
import sys
import math
import seaborn as sns
from matplotlib import pyplot as plt2
import pickle
import time



def taylor_scores(in_dist, out_dist):
    print("test_ood.py ==> taylor_scores()")
    print("np.shape(in_dist): ",np.shape(in_dist) )
    print("np.shape(out_dist): ",np.shape(out_dist) )
    y_true = np.concatenate([np.ones(len(in_dist)), np.zeros(len(out_dist))])
    y_pred = np.concatenate([in_dist, out_dist])
    fpr, fnr, thr = det_curve(y_true, y_pred, pos_label=1)
    det_err = np.min((fnr + fpr) / 2)
    fpr, tpr, thr = roc_curve(y_true, y_pred)
    fpr95_sk = fpr[np.argmax(tpr >= .95)]
    scores = pd.Series({
        "FPR (95% TPR)": fpr95_sk,
        "Detection Error": det_err,
        "AUROC": roc_auc_score(y_true, y_pred),
        "AUPR In": average_precision_score(y_true, y_pred, pos_label=1),
        "AUPR Out": average_precision_score(y_true, 1 - y_pred, pos_label=0),
    })
    return scores


def combine_prob(pred_values):
    print("test_ood.py ==> combine_prob()")
    alpha = 0.5
    print("pred_values: ",pred_values)


class FeatureTester: 
    def __init__(self, dataset: str, model: str, feature_model, name=""):
        print("\n\n-------------test_ood.py ==> FeatureTester-------------")
        self.dataset = dataset
        self.model = model
        # data.img_shape = (32, 32, 3)
        data.img_shape = (224, 224, 3)
        self.data = data.load_dataset(dataset)  # type(self.data) = dict type
        
        print(self.data.keys())
        if "Train" in self.data.keys():
            print(type(self.data["Train"]))

        # self.data["Train"] = self.data["Train"].iloc[:100, :]
        # self.data["Val"] = self.data["Val"].iloc[:100, :]
        # self.data["Test"] = self.data["Test"].iloc[:100, :]

        self.testset_data = self.data["Test"]
        m, transform = load_model(dataset, model)
        # print("load_model : ", m)
        self.path = Path(f"results/{dataset}_{model}")
        self.path = (self.path / name) if name else self.path
        self.path.mkdir(exist_ok=True, parents=True)
        
        # print("Creating Confidenciator", flush=True)
        # print(type(self.data["Train"]))
        self.conf = Confidenciator(m, transform, self.data["Train"])
        # self.conf.plot_model(self.path) TODO implement this.

        print("\n\n   ##  Adding Feature Columns   ##  ")
        print("feature_model :", feature_model)
        for name, df in self.data.items():  
            if feature_model == "mahala":
                print("It is goign in mahala")
                print("running set  :",name)
                self.data[name] = self.conf.add_prediction_and_features_dl(
                    self.data[name]) # name = Train, Test, Val
            else:
                print("ELSE PART IS GETTING EXECUTED")
                print("running set  :",name)
                self.data[name] = self.conf.add_prediction_and_features_knn(
                    self.data[name])
            #self.data[name] = self.conf.add_prediction_and_features(self.data[name], isOOD=False)
            print("Data Frame 2 shape: ", self.data[name].shape)
        print("flag 3 self.data.keys() :", self.data.keys())
        
        # self.compute_accuracy(self.data)
        
        print("\n\n  ##  Creating Out-Of-Distribution Sets  ##  ", flush=True)
        if feature_model == "mahala":
            print("It is goign in mahala")
            self.ood = {name: self.conf.add_prediction_and_features_dl(
                df) for name, df in out_of_dist(self.dataset).items()}
        else:
            print("It is goign in knn")
            self.ood = {name: self.conf.add_prediction_and_features_knn(
                df) for name, df in out_of_dist(self.dataset).items()}
        #self.ood = {name: self.conf.add_prediction_and_features(df) for name, df in out_of_dist(self.dataset).items()}
        #self.ood = {name: df for name, df in out_of_dist(self.dataset).items()}
        print("Length of ood: ", self.ood.keys())
        # self.cal = None  # Training set for the logistic regression.

    def compute_accuracy(self, datasets):
        print("test_ood.py ==> FeatureTester.compute_accuracy()")
        try:
            accuracy = pd.read_csv(
                self.path / "accuracy.txt", sep=":", index_col=0)["Accuracy"]
        except FileNotFoundError:
            accuracy = pd.Series(name="Accuracy", dtype=float)
        for name, df in datasets.items():
            accuracy[name] = df["is_correct"].mean()
            print(f"Accuracy {name}: {accuracy[name]}")
        accuracy.sort_values(ascending=False).to_csv(
            self.path / "accuracy.txt", sep=":")
        print("Done", flush=True)

    def create_summary_combine(self, f, name="", corr=False):
        print("\n\ntest_ood.py ==> FeatureTester.create_summary_combine()")
        print("Creating Taylor Table", flush=True)
        # print(self.ood.keys())
        # for f in ["ft.conf.predict_mahala", "ft.conf.predict_knn"]:
            
            
        pred = {name: f(df) for name, df in self.ood.items()}
        #    pred_values[f] = pred
        #    pred_clean = f(self.data["Test"], isInTest=True)
        #    pred_clean_values[f] = pred_clean
        #pred_clean = f(self.testset_data, isInTest=True)
        
        pred_clean = f(self.data["Test"])
        return pred, pred_clean

    def taylor_table(self, pred, pred_clean, name, method_name, corr=False):
        print("\n\n test_ood.py ==> FeatureTester.taylor_table()")

        all = np.concatenate(list(pred.values()) + [pred_clean])
        print("all :", all)
        p_min, p_max = np.min(all), np.max(all)

        # This function is used since some scores only support values between 0 and 1.
        def map_pred(x):
            print("test_ood.py ==> map_pred()")
            return (x - p_min) / (p_max - p_min)

        pred["All"] = np.concatenate(list(pred.values()))
        print("Until Taylor table everything is good")
        
        table = pd.DataFrame.from_dict(
            {name: taylor_scores(map_pred(pred_clean), map_pred(p)) for name, p in pred.items()}, orient="index")
        
        table.to_csv(self.path / f"summary_{name}.csv")
        df_to_pdf(table, decimals=4, path=self.path /
                  f"summary_{name}.pdf", vmin=0, percent=True)
        self.hist_plot(pred, pred_clean, method_name)
        if corr:
            pred_corr = pred_clean[self.data["Test"]["is_correct"]]
            table = pd.DataFrame.from_dict(
                {name: taylor_scores(map_pred(pred_corr), map_pred(p)) for name, p in pred.items()}, orient="index")
            table.to_csv(self.path / f"summary_correct_{name}.csv")
            df_to_pdf(table, decimals=4, path=self.path /
                      f"summary_correct_{name}.pdf", vmin=0, percent=True)
            

    def create_summary(self, f, name="", corr=False):
        print("test_ood.py ==> FeatureTester.create_summary()")
        print("Creating Taylor Table", flush=True)
        print(self.ood.keys())
        # for f in ["ft.conf.predict_mahala", "ft.conf.predict_knn"]:
        #   pred = {name: f(df) for name, df in self.ood.items()}
        #  pred_values[f] = pred
        #    pred_clean = f(self.data["Test"], isInTest=True)
        #    pred_clean_values[f] = pred_clean
        #pred_clean = f(self.testset_data, isInTest=True)
        #pred_clean = f(self.data["Test"], isInTest=True)

        all = np.concatenate(list(pred.values()) + [pred_clean])
        print(all)
        p_min, p_max = np.min(all), np.max(all)

        # This function is used since some scores only support values between 0 and 1.
        def map_pred(x):
            print("test_ood.py ==> taylor_scores()")
            return (x - p_min) / (p_max - p_min)

        #pred["All"] = np.concatenate(list(pred.values()))
        print("Until Taylor table everything is good")
        table = pd.DataFrame.from_dict(
            {name: taylor_scores(map_pred(pred_clean), map_pred(p)) for name, p in pred.items()}, orient="index")
        table.to_csv(self.path / f"summary_{name}.csv")
        df_to_pdf(table, decimals=4, path=self.path /
                  f"summary_{name}.pdf", vmin=0, percent=True)
        if corr:
            pred_corr = pred_clean[self.data["Test"]["is_correct"]]
            table = pd.DataFrame.from_dict(
                {name: taylor_scores(map_pred(pred_corr), map_pred(p)) for name, p in pred.items()}, orient="index")
            table.to_csv(self.path / f"summary_correct_{name}.csv")
            df_to_pdf(table, decimals=4, path=self.path /
                      f"summary_correct_{name}.pdf", vmin=0, percent=True)

    def test_separation(self, test_set: pd.DataFrame, datasets: dict, name: str, split=False):
        print("test_ood.py ==> FeatureTester.test_separation()")
        if "All" not in datasets.keys():
            datasets["All"] = pd.concat(
                datasets.values()).reset_index(drop=True)
        summary_path = self.path / (f"{name}_split" if split else name)
        summary_path.mkdir(exist_ok=True, parents=True)
        summary = {dataset: {} for dataset in datasets.keys()}
        for feat in np.unique([c.split("_")[0] for c in self.conf.feat_cols]):
            feat_list = [f for f in self.conf.feat_cols if feat in f]
            if split & (feat != "Conf"):
                feat_list = list(
                    sorted([f + "-" for f in feat_list] + [f + "+" for f in feat_list]))
            fig, axs = plt.subplots(len(datasets), len(feat_list), squeeze=False,
                                    figsize=(2 * len(feat_list) + 3, 2.5 * len(datasets)), sharex="col")
            for i, (dataset_name, dataset) in enumerate(datasets.items()):
                if dataset_name != "Clean":
                    dataset = pd.concat([dataset, test_set]).reset_index()
                feats = pd.DataFrame(self.conf.pt.transform(
                    self.conf.scaler.transform(dataset[self.conf.feat_cols])), columns=self.conf.feat_cols)
                if split:
                    cols = list(feats.columns)
                    feats = pd.DataFrame(split_features(feats.to_numpy()),
                                         columns=[c + "+" for c in cols] + [c + "-" for c in cols])
                for j, feat_id in enumerate(feat_list):
                    summary[dataset_name][feat_id] = binary_class_hist(feats[feat_id], dataset["is_correct"],
                                                                       axs[i, j], "", bins=50,
                                                                       label_1="ID", label_0=dataset_name)
            for ax, col in zip(axs[0], feat_list):
                ax.set_title(f"Layer {col}")

            for ax, row in zip(axs[:, 0], datasets.keys()):
                ax.set_ylabel(row, size='large')
            plt.tight_layout(pad=.4)
            plt.savefig(summary_path / f"{feat}.pdf")
        if split:
            summary["LogReg Coeff"] = self.conf.coeff
        # save_corr_table(feature_table, self.path / f"corr_distorted", self.dataset_name)
        summary = pd.DataFrame(summary)
        summary.to_csv(f"{summary_path}.csv")
        df_to_pdf(summary, decimals=4,
                  path=f"{summary_path}.pdf", vmin=0, percent=True)

    def fit_knn(self, test: bool, c=None):
        print("test_ood.py ==> FeatureTester.fit_knn()")
        """if test:
            #self.cal = {value : self.data[i] for i in ["Train", "Val"]}
            self.cal = self.load_cal_data()
            print("Keys: ", self.cal.keys())
            print("Fitting KNN ", flush=True)
            self.conf.fit_knn(self.cal, c=c)
        else:
            if not self.cal:
                print("Creating Calibration Set", flush=True)
                self.cal = calibration(self.data["Val"])
            print("Fitting KNN Regression", flush=True)"""
        self.conf.fit_knn_faiss(self.data["Train"], c=c)

    def fit(self, c=None, new_cal_set=False):
        print("test_ood.py ==> FeatureTester.fit()")
        if new_cal_set or not self.cal:
            print("Creating Calibration Set", flush=True)
            self.cal = calibration(self.data["Val"])
        print("Fitting Logistic Regression", flush=True)
        self.conf.fit(self.cal, c=c)

    def test_ood(self, split=False):
        print("test_ood.py ==> FeatureTester.test_ood()")
        print("\n==================   Testing features on Out-Of-Distribution Data   ==================\n",
              flush=True)
        self.test_separation(self.data["Test"].assign(
            is_correct=True), self.ood, "out_of_distribution", split)

    def test_distorted(self, split=False):
        print("test_ood.py ==> FeatureTester.test_distorted()")
        print("\n=====================   Testing features on Distorted Data   =====================\n", flush=True)
        dist = distorted(self.data["Test"])
        dist = {name: self.conf.add_prediction_and_features(
            df) for name, df in dist.items()}
        self.compute_accuracy(dist)
        self.test_separation(self.data["Test"], dist, "distorted", split)

    def plot_detection(self, f, name):
        print("test_ood.py ==> FeatureTester.plot_detection()")
        path = self.path / f"detection/{name}"
        path.mkdir(exist_ok=True, parents=True)
        pred = {name: f(df) for name, df in self.ood.items()}
        pred_clean = f(self.data["Test"])
        plt.figure(figsize=(4, 3))
        for key, p in pred.items():
            plt.clf()
            labels = pd.Series(np.concatenate(
                [np.ones(len(pred_clean), dtype=bool), np.zeros(len(p), dtype=bool)]))
            p = pd.Series(np.concatenate([pred_clean, p]))
            binary_class_hist(p, labels, plt.gca(), name,
                              label_0="OOD", label_1="ID")
            plt.tight_layout()
            plt.savefig(path / f"{key}.pdf")
            
    def hist_plot(self, pred_ood, pred_clean, method_name):
        if isinstance(pred_ood, dict):
          ood_df = {name: pd.DataFrame(-df, columns=[name]) for name, df in pred_ood.items()}
          print("this passes")
          in_df = pd.DataFrame(-pred_clean, columns=["clean"])
          result = {name: pd.concat([df, in_df]) for name, df in ood_df.items()}
          plt2.figure(figsize=(4, 3))
          for key, value in result.items():
              plt2.clf()
              sns.histplot(data=result[key])
              plt2.savefig(self.path / f"save_histogram_{method_name}_{str(key)}.png")

        return result

def log_probability(pred_mahala, pred_knn):
    print("test_ood.py ==> log_probability()")
    n = 2048 #512
    if isinstance(pred_mahala, dict):
        result = {name: ((-n * (np.log(-(pred_knn[name])))) - (pred_mahala[name] ** 2)) for name, df in pred_mahala.items()}
    else:
        result = ((-n * (np.log(-(pred_knn)))) - (pred_mahala ** 2))
    return result

def normalized_log_probability(pred_mahala, pred_knn, mahala_mean, knn_mean, mahala_std, knn_std):
    n = 2048 #512
    if isinstance(pred_mahala, dict):
        pred_mahala_result = {name: - (pred_mahala[name] ** 2) for name, df in pred_mahala.items()}
        pred_knn_result = {name: (-n * np.log(-pred_knn[name])) for name, df in pred_mahala.items()}

        result = {name: (((pred_mahala_result[name] - mahala_mean) / mahala_std) 
            + (pred_knn_result[name] - knn_mean) / knn_std ) for name, df in pred_mahala.items()}
    else:
        pred_knn_result = -n * np.log(-pred_knn)
        pred_mahala_result = - (pred_mahala **2)

        result = ((pred_mahala_result - mahala_mean) / mahala_std + (pred_knn_result - knn_mean) / knn_std)
    return result

def square_log_probability(pred_mahala, pred_knn):
    print("test_ood.py ==> square_log_probability()")
    n = 512
    if isinstance(pred_mahala, dict):
        result = {name: ((- math.sqrt(n) * (np.log(-(pred_knn[name])))) - (pred_mahala[name] ** 2)) for name, df in pred_mahala.items()}
    else:
        result = ((- (math.sqrt(n)) * (np.log(-(pred_knn)))) - (pred_mahala ** 2))
    return result

def weighted_geometric_mean(pred_mahala, pred_knn, alpha):
    print("test_ood.py ==> weighted_geometric_mean()")
    if isinstance(pred_mahala, dict):
        result = {name: -(((-df) ** alpha) * ((-pred_knn[name]) ** (
            1 - alpha))) for name, df in pred_mahala.items()}
    else:
        result = -(((-pred_mahala) ** alpha) * ((-pred_knn) ** (1-alpha)))
    return result

def weighted_arthmetic_mean(pred_mahala, pred_knn, mahala_mean, knn_mean, alpha):
    print("test_ood.py ==> weighted_arthmetic_mean()")
    #mean_mahala = {name: df.mean() for name, df in pred_mahala.items()}
    #mean_knn = {name: df.mean() for name, df in pred_knn.items()}
    #norm_pred_mahala = {name: df - mean_mahala[name] for name, df in pred_mahala.items()}
    #norm_pred_knn = {name: df - mean_knn[name] for name, df in pred_knn.items()}
    if isinstance(pred_mahala, dict):
        result = {name: -(alpha * (-df/mahala_mean) + (1-alpha) * (-pred_knn[name]/knn_mean)) for name, df in pred_mahala.items()}
    else:
        result = -((alpha) * (-pred_mahala/mahala_mean) + (1-alpha) * (-pred_knn/knn_mean))
    return result

def max_distance(pred_mahala, pred_knn, mahala_mean, knn_mean, mahala_std, knn_std):
    n = 512
    if isinstance(pred_mahala, dict):
        
        result = {name: np.maximum((((pred_mahala[name]) - mahala_mean) / mahala_std), 
            (((pred_knn[name]) - knn_mean) / knn_std)) for name, df in pred_mahala.items()}
    else:
        result = np.maximum((((pred_mahala) - mahala_mean) / mahala_std), 
            (((pred_knn) - knn_mean) / knn_std))
    return result

def max_distance2(pred_mahala, pred_knn, mahala_mean, knn_mean, mahala_std, knn_std):
    n = 512
    if isinstance(pred_mahala, dict):
        
        result = {name: np.maximum((np.abs(pred_mahala[name] - mahala_mean) / mahala_std), 
            (np.abs(pred_knn[name] - knn_mean) / knn_std)) for name, df in pred_mahala.items()}
    else:
        result = np.maximum((np.abs(pred_mahala - mahala_mean) / mahala_std), 
            (np.abs(pred_knn - knn_mean) / knn_std))
    return result

def hist_plot_mahala_knn(pred_mahala, pred_knn, method_name):
    if isinstance(pred_mahala, dict):
      mahala_df = {name: pd.DataFrame(-df, columns=["mahala_" + name]) for name, df in pred_mahala.items()}
      knn_df = {name: pd.DataFrame(-df, columns=["knn_" + name]) for name, df in pred_knn.items()}
      result = {name: pd.concat([mahala_df[name], knn_df[name]]) for name, df in pred_knn.items()}
      for key, value in result.items():
        sns.histplot(data=result[key])
        plt.savefig(f"save_histogram_{method_name}_{str(key)}.png")
    return result


def test_ood(dataset, model, alpha):
    print("test_ood.py ==> test_ood()")
    print(
        f"\n\n================ Testing Features On {dataset} {model} ================", flush=True)
    pred_probs = []
    pred_clean_probs = []
    #ft.create_summary(ft.conf.predict_mahala, "x-ood-mahala")

    # ft_mahala
    ft_mahala = FeatureTester(dataset, model, "mahala", "")
    pred_mahala, pred_clean_mahala = ft_mahala.create_summary_combine(
        ft_mahala.conf.predict_mahala, "x-ood-mahala")
    ft_mahala.taylor_table(pred_mahala, pred_clean_mahala,
                            "x-ood-mahala-" + str(alpha), "mahala")

    #
    
    # ft_knn
    ft_knn = FeatureTester(dataset, model, "", "knn")
    ft_knn.fit_knn(test=False)
    pred_knn, pred_clean_knn = ft_knn.create_summary_combine(
        ft_knn.conf.predict_knn_faiss, "open-ood-knn")
    ft_knn.taylor_table(pred_knn, pred_clean_knn, "knn-" + str(alpha), "knn")
    # hist_plot_mahala_knn(pred_mahala,pred_knn,"mahala_knn")
    
    # if dataset == "imagenet":
    #     filename = "train_"+ str(dataset) + "_" + str(model) + "_"
    # else:
    #     filename = str(dataset) + "_" + str(model) + "_"
    # with open('/home/saiful/confidence-magesh_MR/confidence-magesh/OpenOOD/pickle_files/'+filename+'_pred_knn.pickle', 'wb') as handle1:
    #     pickle.dump(pred_knn, handle1, protocol=pickle.HIGHEST_PROTOCOL)
    # with open('/home/saiful/confidence-magesh_MR/confidence-magesh/OpenOOD/pickle_files/'+filename+'_pred_clean_knn.pickle', 'wb') as handle2:
    #     pickle.dump(pred_clean_knn, handle2, protocol=pickle.HIGHEST_PROTOCOL)
    # with open('/home/saiful/confidence-magesh_MR/confidence-magesh/OpenOOD/pickle_files/'+filename+'_pred_mahala.pickle', 'wb') as handle3:
    #     pickle.dump(pred_mahala, handle3, protocol=pickle.HIGHEST_PROTOCOL)
    # with open('/home/saiful/confidence-magesh_MR/confidence-magesh/OpenOOD/pickle_files/'+filename+'pred_clean_mahala.pickle', 'wb') as handle4:
    #     pickle.dump(pred_clean_mahala, handle4, protocol=pickle.HIGHEST_PROTOCOL)
    
    # # # commenting for concatenated- untill f_ft_knn.taylor table
    # # pred_probs.append(pred_knn) 
    # # pred_clean_probs.append(pred_clean_knn)
    
    # weighted_arthmetic_mean
    pred_arth = weighted_arthmetic_mean(pred_mahala, pred_knn, ft_mahala.conf.mahala_mean, ft_knn.conf.knn_mean, alpha)
    print("pred_arth Keys:", pred_arth.keys())
    pred_clean_arth = weighted_arthmetic_mean(pred_clean_mahala, pred_clean_knn, ft_mahala.conf.mahala_mean, ft_knn.conf.knn_mean, alpha)
    ft_knn.taylor_table(pred_arth, pred_clean_arth, "x-ood-mahala-knn-arth-" + str(alpha),"arthmetic_mean")

    # weighted_geometric_mean
    pred_geo = weighted_geometric_mean(pred_mahala, pred_knn, alpha)
    pred_clean_geo = weighted_geometric_mean(
        pred_clean_mahala, pred_clean_knn, alpha)
    ft_knn.taylor_table(pred_geo, pred_clean_geo, "x-ood-mahala-knn-geo-" + str(alpha),"geometric_mean")
   
 
    # log probabilty
    pred_log = log_probability(pred_mahala, pred_knn)
    pred_clean_log = log_probability(pred_clean_mahala, pred_clean_knn)
    ft_knn.taylor_table(pred_log, pred_clean_log, "x-ood-mahala-knn-log", "log_probability" )
    
    # square log probabilty  
    pred_sq_log = square_log_probability(pred_mahala, pred_knn)
    pred_clean_sq_log = square_log_probability(pred_clean_mahala, pred_clean_knn)
    ft_knn.taylor_table(pred_sq_log, pred_clean_sq_log, "x-ood-mahala-knn-log-sq", "square_log_probability")
    
    # normalized_log_probability
    pred_n_log = normalized_log_probability(pred_mahala, pred_knn,
            ft_mahala.conf.mahala_mean, ft_knn.conf.knn_mean, ft_mahala.conf.mahala_std, ft_knn.conf.knn_std)
    pred_n_clean_log = normalized_log_probability(pred_clean_mahala, pred_clean_knn, 
            ft_mahala.conf.mahala_mean, ft_knn.conf.knn_mean, ft_mahala.conf.mahala_std, ft_knn.conf.knn_std)
    ft_knn.taylor_table(pred_n_log, pred_n_clean_log, "x-ood-mahala-knn-n-log","normalized_log_probability")
    
    # max_distance
    pred_max = max_distance(pred_mahala, pred_knn, 
            ft_mahala.conf.mahala_max_mean, ft_knn.conf.knn_max_mean, ft_mahala.conf.mahala_max_std, ft_knn.conf.knn_max_std)
    pred_clean_max = max_distance(pred_clean_mahala, pred_clean_knn, 
            ft_mahala.conf.mahala_max_mean, ft_knn.conf.knn_max_mean, ft_mahala.conf.mahala_max_std, ft_knn.conf.knn_max_std)
    ft_knn.taylor_table(pred_max, pred_clean_max, "x-ood-mahala-knn-max-","mahala_max_mean" )


    # ft_mahala.create_summary_combine(ft_mahala.conf.softmax, "baseline")
    # ft.create_summary(ft.conf.energy, "energy")
    # ft.create_summary(ft.conf.react_energy, "react_energy")
    # for i in range(10):
    # ft.fit(new_cal_set=True)
    #ft.create_summary(ft.conf.predict_proba, f"x-ood-lr")
    # ft.fit_knn(test=False)
    #ft.create_summary(ft.conf.predict_knn, f"x-ood-knn")
    # ft.fit_knn_faiss()
    # Add KNN Faiss algorithm to this
    #ft.create_summary(ft.conf.predict_knn_faiss, f"knn-open-ood")
    # ft.test_distorted()
    # ft.test_ood()


if __name__ == "__main__":
    
    start_time = time.time()
    
    # sys.stdout = open("console_output_knn.txt", "w")
    # test_ood("mnist", "lenet", 0.5)
    # test_ood("cifar10", "resnet", 0.5)
    test_ood("cifar100", "resnet", 0.5)
    # test_ood("imagenet", "resnet50", 0.5)
    # for i in [0.7]:
    #   test_ood("imagenet", "resnet34", i)
    #   test_ood("cifar10", "resnet", i)
    #  test_ood("cifar100", "resnet", i)
    #test_ood("cifar100", "resnet", 0.7)
    #test_ood("cifar100", "resnet50")
    #test_ood("cifar100", "resnet101")
    # for m in "resnet", "densenet":
    # for m in "densenet":
    #   for d in "svhn", "cifar10", "cifar100":
    #      test_ood(d, m)
    # for m in "resnet18", "resnet34", "resnet50", "resnet101":
    #     test_ood("imagenet", m)
    print("\nExecution Complete")
    print("--- %s seconds ---" % (time.time() - start_time))
