import torch.optim as optim

import torch.nn as nn

from dataset import load_ddi_dataset
from logger.train_logger import TrainLogger
from data_pre import CustomData
import argparse
from metrics import *
from utils import *
from tqdm import tqdm
import warnings

from model import gnn_model
warnings.filterwarnings("ignore")

def val(SRR, criterion, dataloader, device, epoch):
    SRR.eval()
    running_loss = AverageMeter()

    pred_list = []
    label_list = []

    for data in tqdm(dataloader,desc='val_epoch_{}'.format(epoch),leave=True):
        head_pairs, tail_pairs, head_pairs_dgl, tail_pairs_dgl, rel, label = [d.to(device) for d in data]
        sim_h = head_pairs.sim
        sim_t = tail_pairs.sim

        batch_h_e = head_pairs_dgl.edata['feat']
        batch_t_e = tail_pairs_dgl.edata['feat']
        with torch.no_grad():
            pred = SRR.forward(head_pairs, tail_pairs, head_pairs_dgl, tail_pairs_dgl,batch_h_e, batch_t_e, rel, sim_h, sim_t)
            loss = criterion(pred, label)
            pred_cls = torch.sigmoid(pred)
            pred_list.append(pred_cls.view(-1).detach().cpu().numpy())
            label_list.append(label.detach().cpu().numpy())
            running_loss.update(loss.item(), label.size(0))

    pred_probs = np.concatenate(pred_list, axis=0)
    label = np.concatenate(label_list, axis=0)

    acc, auroc, f1_score, precision, recall, ap = do_compute_metrics(pred_probs, label)

    epoch_loss = running_loss.get_average()
    running_loss.reset()

    SRR.train()

    return epoch_loss, acc, auroc, f1_score, precision, recall, ap

def main():
    parser = argparse.ArgumentParser()

    # Add argument
    parser.add_argument('--n_iter', type=int, default=10, help='number of GNN')
    parser.add_argument('--L', type=int, default=3, help='number of Graph Transformer')
    parser.add_argument('--fold', type=int, default=1, help='[0, 1, 2]')
    parser.add_argument('--epochs', type=int, default=200, help='number of epochs')
    parser.add_argument('--weight_decay', type=float, default=5e-4, help='number of epochs')
    parser.add_argument('--batch_size', type=int, default=512, help='batch size')
    parser.add_argument('--save_model', action='store_true', help='whether save model or not')
    parser.add_argument('--lr', type=float, default=1e-3, help='learning rate')
    args = parser.parse_args()

    params = dict(
        model='SRR-DDI',
        data_root='/tmp/SRR-DDI/DrugBank/data/warm start',
        save_dir='save',
        dataset='drugbank',
        epochs=args.epochs,
        fold=args.fold,
        save_model=args.save_model,
        lr=args.lr,
        batch_size=args.batch_size,
        n_iter=args.n_iter,
        weight_decay=args.weight_decay,
        L = args.L
    )

    logger = TrainLogger(params)
    logger.info(__file__)

    save_model = params.get('save_model')
    batch_size = params.get('batch_size')
    data_root = params.get('data_root')
    data_set = params.get('dataset')
    fold = params.get('fold')
    epochs = params.get('epochs')
    n_iter = params.get('n_iter')
    lr = params.get('lr')
    L = params.get('L')
    weight_decay = params.get('weight_decay')
    data_path = os.path.join(data_root, data_set)

    train_loader, val_loader, test_loader = load_ddi_dataset(root=data_path, batch_size=batch_size, fold=fold)
    data = next(iter(train_loader))
    node_dim = data[0].x.size(-1)
    edge_dim = data[0].edge_attr.size(-1)
    device = torch.device('cuda:1' if torch.cuda.is_available() else "cpu")

    MODEL_NAME = 'GraphTransformer'
    net_params = dict(
        L=3,
        n_heads=8,
        hidden_dim=128,
        out_dim=128,
        edge_feat=True,
        residual=True,
        readout="mean",
        in_feat_dropout=0.0,
        dropout=0.0,
        layer_norm=False,
        batch_norm=True,
        self_loop=False,
        lap_pos_enc=True,
        pos_enc_dim=6,
        full_graph=False,
        batch_size=512,
        num_atom_type=node_dim,
        num_bond_type=edge_dim,
        device=device,
        n_iter=n_iter
    )
    print("start with warm start fold_{}".format(fold) + " transformer have {} layers".format(L),"sub extract have {} layers".format(n_iter))
    SRR = gnn_model(MODEL_NAME, net_params).to(device)

    optimizer = optim.Adam(SRR.parameters(), lr=lr, weight_decay=weight_decay)
    criterion = nn.BCEWithLogitsLoss()
    scheduler = optim.lr_scheduler.LambdaLR(optimizer, lambda epoch: 0.96 ** (epoch))

    running_loss = AverageMeter()
    running_acc = AverageMeter()


    SRR.train()
    for epoch in range(epochs):
        for data in tqdm(train_loader,desc='train_loader_epoch_{}'.format(epoch),leave=True):

            head_pairs, tail_pairs, head_pairs_dgl, tail_pairs_dgl, rel, label = [d.to(device) for d in data]

            sim_h = head_pairs.sim
            sim_t = tail_pairs.sim

            batch_h_e = head_pairs_dgl.edata['feat']
            batch_t_e = tail_pairs_dgl.edata['feat']
            pred = SRR.forward(head_pairs, tail_pairs, head_pairs_dgl, tail_pairs_dgl,batch_h_e, batch_t_e, rel, sim_h, sim_t)
            loss = criterion(pred, label)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            pred_cls = (torch.sigmoid(pred) > 0.5).detach().cpu().numpy()
            acc = accuracy(label.detach().cpu().numpy(), pred_cls)
            running_acc.update(acc)
            running_loss.update(loss.item(), label.size(0))

        epoch_loss = running_loss.get_average()
        epoch_acc = running_acc.get_average()
        running_loss.reset()
        running_acc.reset()

        val_loss, val_acc, val_auroc, val_f1_score, val_precision, val_recall, val_ap = val(SRR, criterion, val_loader, device, epoch)
        val_msg = "epoch-%d, train_loss-%.4f, train_acc-%.4f, val_loss-%.4f, val_acc-%.4f, val_auroc-%.4f, val_f1_score-%.4f, val_prec-%.4f, val_rec-%.4f, val_ap-%.4f" % (epoch, epoch_loss, epoch_acc, val_loss, val_acc, val_auroc, val_f1_score, val_precision, val_recall, val_ap)

        logger.info(val_msg)
        scheduler.step()
        if save_model:
            msg = "epoch-%d, train_loss-%.4f, train_acc-%.4f, val_loss-%.4f, val_acc-%.4f" % (epoch, epoch_loss, epoch_acc, val_loss, val_acc)
            save_model_dict(SRR, logger.get_model_dir(), msg)


if __name__ == "__main__":
    main()




