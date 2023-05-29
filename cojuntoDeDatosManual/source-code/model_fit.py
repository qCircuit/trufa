import mlflow
import numpy as np
import pandas as pd
import seaborn as sns
import time
import torch
from scipy.spatial.distance import jensenshannon
from sklearn.metrics import log_loss

import configura
import utilities

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

class MyDataset(torch.utils.data.Dataset):   
    def __init__(self, x, y):
        self.x = x
        self.y = y
        
    def __getitem__(self, index):
        return self.x[index], self.y[index]
    
    def __len__(self):
        return len(self.x)
    
def get_loader(df, shuffle_bool=False):
    data = MyDataset(
        torch.FloatTensor(df.drop(configura.target.keys(), axis=1).to_numpy()),
        torch.FloatTensor(df[configura.target.keys()].to_numpy()),
    )
    loader = torch.utils.data.DataLoader(
        dataset=data, batch_size=configura.batch_size, shuffle=shuffle_bool
    )

    return loader

def binary_acc(y_pred: np.array, y_test: np.array) -> float:
    y_pred_tag = torch.round(torch.sigmoid(y_pred))

    correct_results_sum = (y_pred_tag == y_test).sum().float()
    acc = correct_results_sum/y_test.shape[0]
    acc = torch.round(acc * 100)
    
    return acc

def torch_fit(nn, train_loader, test_loader): 

    model = nn
    model.to(device)
    criterion = torch.nn.BCEWithLogitsLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=configura.learning_rate)

    stats = {"loss_train":[], "loss_valid":[], "accuracy_train":[], "accuracy_valid":[]}

    for e in range(1, configura.epochs+1):
        epoch_loss, epoch_acc = 0, 0
        valid_loss, valid_acc = 0, 0
        st_tm = time.time()
        model.train()
        for x_batch, y_batch in train_loader:
            x_batch, y_batch = x_batch.to(device), y_batch.to(device)
            optimizer.zero_grad()
            
            y_pred = model(x_batch)
            loss = criterion(y_pred, y_batch)
            acc = binary_acc(y_pred, y_batch)
            
            loss.backward()
            optimizer.step()
            
            epoch_loss += loss.item()
            epoch_acc += acc.item()
        model.eval()
        for x_batch, y_batch in test_loader:
            x_batch, y_batch = x_batch.to(device), y_batch.to(device)

            y_pred = model(x_batch)
            valid_loss = criterion(y_pred, y_batch)
            valid_acc = binary_acc(y_pred, y_batch)

            valid_loss += valid_loss.item()
            valid_acc += valid_acc.item()
        
        stats["loss_train"].append(epoch_loss)
        stats["loss_valid"].append(valid_loss)
        stats["accuracy_train"].append(epoch_acc)
        stats["accuracy_valid"].append(valid_loss)

        utilities.logger.info(
            f"Epoch {e+0:03}/{configura.epochs+0:03}: | Loss: {epoch_loss/len(train_loader):.5f} | Acc: {epoch_acc/len(train_loader):.3f}"\
            f"| Valid_Loss: {valid_loss/len(test_loader):.5f} | Valid_Acc: {valid_acc/len(test_loader):.3f} "\
            f"| Time: {time.time() - st_tm:.1f}"
        )

    torch.save(model.state_dict(), configura.model_path)
    utilities.logger.info("model fit completed")

    return model

def get_preds(model, loader, name):

    probs = []
    for x in loader:
        r = x[0].to(device)
        with torch.no_grad():
            probs.append(torch.sigmoid(model(r.to(device)).detach().cpu()).numpy())    
    probs = [l.squeeze().tolist() for sl in probs for l in sl]
    
    return pd.Series(probs, name=name)

def evaluate_torch_model(model, test_loader):

    model.eval()

    facts = [int(l) for sl in [v[1] for v in iter(test_loader)] for l in sl]
    facts = pd.Series(facts, name=list(configura.target.keys())[0])
    probs = pd.concat([get_preds(model, test_loader, 'r').reset_index(drop=True), facts.reset_index(drop=True)], axis=1)

    n_bins, n_dec, border = configura.test_params
    probs = probs.round({"r":n_dec})
    probs['group'] = pd.cut(
        probs.r, 
        n_bins, 
        right=True,
    ).astype(str)    
    
    # get averages
    probs_to_plot = pd.DataFrame()
    probs_to_plot['r_mean'] = probs.groupby(by='group').mean()['r']
    probs_to_plot['p'] = probs.groupby(by='group').mean()[list(configura.target.keys())]

    # get quantity
    probs_to_plot['n'] = probs.groupby(by='group').count()['r']

    sns.set_theme(style='darkgrid')
    sns.set(rc={'figure.figsize':(13, 9)})
    g = sns.scatterplot(
        x='r_mean', y='p', data=probs_to_plot,
        color = 'darkblue',
        size = probs_to_plot['n'].values
    )
    g.plot(np.linspace(0, border, 100), np.linspace(0, border, 100))
    g.set(xlim=(0, border), ylim=(0, border))

    resulting_metrics = {
        "mae": round(sum(abs(probs_to_plot.p - probs_to_plot.r_mean))/len(probs_to_plot),5),
        "mse": round(sum((probs_to_plot.p - probs_to_plot.r_mean) ** 2)/ len(probs_to_plot),5),
        "jensen_shannon": jensenshannon(probs_to_plot.p , probs_to_plot.r_mean),
        "r_mean": probs.r.mean(),
        "p_mean": facts.mean(),
        "means_positive": probs[probs[list(configura.target.keys())[0]]==1].r.mean(),
        "means_negative": probs[probs[list(configura.target.keys())[0]]==0].r.mean(),
        "prediction_range": probs.r.max()-probs.r.min(),
        "log_loss": log_loss(facts, probs.r)
    }

    mlflow.log_metrics(resulting_metrics)
    probs.to_csv(configura.probabilities_path, index=False)
    g.figure.savefig(configura.test_plot_path)
    utilities.sendMessage(f"idx: {configura.idx}")
    utilities.sendImage(configura.test_plot_path)

    return probs, resulting_metrics