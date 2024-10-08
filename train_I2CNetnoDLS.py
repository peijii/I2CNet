# -*- coding: utf-8 -*-
from sklearn.model_selection import KFold
import torch.optim as optim
from torch.utils.data import DataLoader
from src.models.featureExtractor import *
from src.models.labelPredictor import *
from src.models.model_e_p import *
from src.models.model import *
from src.tools.tools import *
from src.utils.utils import ISRUCS3, ReadConfig
from src.utils.augmentations import *
import argparse
os.environ["CUDA_LAUNCH_BLOCKING"] = "1"


if __name__ == '__main__':

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # 1. Get Configuration
    parser = argparse.ArgumentParser()
    parser.add_argument("-c", type=str, help="configuration file", required=True)
    args = parser.parse_args()
    cfgPath, cfgTrain, cfgModel = ReadConfig(args.c)

    # parameters for training phase
    seed           = int(cfgTrain["seed"])
    batch_size     = int(cfgTrain["batch_size"])
    learning_rate  = float(cfgTrain["learning_rate"])
    num_epochs     = int(cfgTrain["num_epochs"])
    Dataset_path   = str(cfgPath["data_path"])
    Fold           = int(cfgTrain["Fold"])

    # parameters for models
    channel        = int(cfgModel["channel"])
    num_classes    = int(cfgModel["num_classes"])
    mse_b1         = int(cfgModel["mse_b1"])
    mse_b2         = int(cfgModel["mse_b2"])
    mse_b3         = int(cfgModel["mse_b3"])
    expansion_rate = int(cfgModel["expansion_rate"])
    reduction_rate = int(cfgModel["reduction_rate"])
    block1_num     = int(cfgModel["block1_num"])
    block2_num     = int(cfgModel["block2_num"])

    seed_everything(42)
    data_paths = [os.path.join(Dataset_path, i) for i in os.listdir(Dataset_path)]

    torch.cuda.set_device(0)
    kf = KFold(n_splits=Fold, shuffle=True, random_state=42)
    Fold_acc = []
    Fold_f1 = []

    # ======================== step 1/5 Data ==============================
    for fold, (train_idx, test_idx) in enumerate(kf.split(data_paths)):
        print(f"Fold {fold + 1}")

        # Splitting the dataset
        train_paths = np.array(data_paths)[train_idx]
        test_paths = np.array(data_paths)[test_idx]

        # Creating dataset objects for train and test
        train_dataset = ISRUCS3(paths=train_paths, transforms=None)
        test_dataset = ISRUCS3(paths=test_paths, transforms=None)

        # 构建DataLoader
        train_loader = DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=True, num_workers=36)
        test_loader = DataLoader(dataset=test_dataset, batch_size=batch_size, shuffle=False, num_workers=36)

        # ======================== step 2/5 Model ==============================
        Total_model = I2CNet(
                in_planes=channel,
                num_classes=num_classes,
                mse_b1=mse_b1,
                mse_b2=mse_b2,
                mse_b3=mse_b3,
                expansion_rate=expansion_rate,
                reduction_rate=reduction_rate,
                cell1_num=block1_num,
                cell2_num=block2_num
        )

        model_e_p = Total_model.model_e_p
        model_e_p.to(device)

        # ======================== step 3/5 Loss function ==============================
        model_e_p_ce_loss = nn.CrossEntropyLoss()
        # ======================== step 4/5 Optimizers ==============================
        model_e_p_optimizer = optim.AdamW(model_e_p.parameters(), lr=learning_rate, weight_decay=1e-4)
        scheduler = optim.lr_scheduler.MultiStepLR(model_e_p_optimizer, gamma=0.8, milestones=[20, 40, 60, 80])

        # ======================== step 5/5 Train ==============================
        loss_rec = {"trian": [], "valid": []}
        acc_rec = {"train": [], "valid": []}
        best_acc, best_f1, best_epoch = 0, 0, 0

        for epoch in range(num_epochs):

            # train
            loss_train, acc_train, f1_train = ModelTrainer.train(
                data_loader=train_loader,
                model=model_e_p,
                loss_f=model_e_p_ce_loss,
                optimizer=model_e_p_optimizer,
                epoch_id=epoch,
                device=device,
                max_epoch=num_epochs,
                num_classes=num_classes
            )
            # valid
            loss_val, acc_valid, f1_valid = ModelTrainer.valid(
                data_loader=test_loader,
                model=model_e_p,
                loss_f=model_e_p_ce_loss,
                device=device,
                num_classes=num_classes
            )

            if acc_valid > best_acc:
                best_acc = acc_valid
                best_f1 = f1_valid

            print("Epoch[{:0>3}/{:0>3}] Train Acc: {:.2%} Valid Acc: {:.2%} "
                  "Train loss:{:.4f} Valid loss:{:.4f} "
                  "Train F1: {:.2%} Valid F1: {:.2%} "
                  "Best Acc: {:.2%} Best F1: {:.2%}".format(
                epoch+1, num_epochs, acc_train, acc_valid, loss_train, loss_val, f1_train, f1_valid, best_acc, best_f1
            ))

            scheduler.step()

        Fold_acc.append(best_acc)
        Fold_f1.append(best_f1)

    print(128 * "=")
    print(Fold_acc)
    print(Fold_f1)
    print(128 * "=")


