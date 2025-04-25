import argparse
import os
import random
import sys

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import yaml
import models
from datasets.dataloader import ImageQualityDataLoader
from scipy import stats


class IQAManager(object):
    def __init__(self, options, path, percentage, rd):
        """Prepare the network, criterion, solver, and data.
        Args:
            options, dict: Hyperparameters.
        """
        print('Prepare the network and data.')
        self._options = options
        self._path = path
        self.percentage = percentage
        self.round = rd

        # Network.
        self._net, self.cfg = models.buildModel(options['model'], options['cfgname'])

        # Criterion.
        self._criterion = nn.MSELoss().cuda()

        # Solver.
        self._solver = optim.Adam(
            [
                {'params': self._net.prompt_learner_aux.parameters()},
                {'params': self._net.prompt_learner_tar.parameters()},
                {'params': self._net.image_residual_block.parameters()}
            ],
            lr=self._options['base_lr']
        )

        dn = self._options['dataset']

        # Data loaders.
        self._train_loader = ImageQualityDataLoader(
            dn, self._path[dn], self._options['train_index'],
            batch_size=self._options['batch_size'], istrain=True, patch_num=1
        ).get_data()

        self._test_loader = ImageQualityDataLoader(
            dn, self._path[dn], self._options['test_index'],
            istrain=False
        ).get_data()

        self.unfold = nn.Unfold(kernel_size=(224, 224), stride=64)

        # Save paths.
        save_path = os.path.join(
            './save/', 'aux', options['dataset'], f'round{options["round"]}',
            options['model'],
            f'use_aux_{self.cfg["TRAINER"]["Ours"]["use_aux"]}_textdepth_{self.cfg["TRAINER"]["Ours"]["text_depth"]}_visiondepth_{self.cfg["TRAINER"]["Ours"]["vision_depth"]}_lamda_{self.cfg["TRAINER"]["Ours"]["lamda"] * 100}'
        )
        os.makedirs(save_path, exist_ok=True)

        options['savePath'] = save_path

        self.savePath_aux = os.path.join(options['savePath'],
                                         f'{options["model"]}_{options["dataset"]}_{options["n_ctx"]}_aux_best.pth')
        self.savePath_tar = os.path.join(options['savePath'],
                                         f'{options["model"]}_{options["dataset"]}_{options["n_ctx"]}_tar_best.pth')
        self.testDataPath = os.path.join(options['savePath'], f'{options["model"]}_{options["dataset"]}_best')

    def train(self):
        """Train the network."""
        print('Training.')
        best_srcc, best_plcc, best_krcc, best_epoch = 0.0, 0.0, 0.0, None
        not_continue_count = 0

        print('Epoch\tTrain Loss\tAux Loss\tTrain SRCC\tTest SRCC\tTest PLCC\tTest KRCC')
        for t in range(self._options['epochs']):
            epoch_loss, epoch_loss1 = [], []
            pscores, tscores = [], []

            for X, y, z, _, y1 in self._train_loader:
                X, y, z, y1 = X.cuda(), y.cuda().float(), z.cuda(), y1.cuda().float()
                score, aligend_score = self._net(X, z)

                self._solver.zero_grad()
                loss = self._criterion(score, y.view(len(score), 1).detach())
                aux_loss = self._criterion(aligend_score, y1.view(len(score), 1).detach()) if \
                    self.cfg['TRAINER']['Ours']['use_aux'] else torch.zeros(1).to(loss.device)

                epoch_loss.append(loss.item())
                epoch_loss1.append(aux_loss.item())
                pscores.extend(score.cpu().tolist())
                tscores.extend(y.cpu().tolist())

                loss.backward()
                self._solver.step()

            train_srcc, _ = stats.spearmanr(pscores, tscores)
            test_srcc, test_plcc, test_data, test_krcc = self.test(self._test_loader)

            if test_srcc > best_srcc:
                best_srcc, best_plcc, best_krcc, best_epoch = test_srcc, test_plcc, test_krcc, t + 1
                torch.save(self._net.prompt_learner_aux.state_dict(), self.savePath_aux)
                torch.save(self._net.prompt_learner_tar.state_dict(), self.savePath_tar)
                np.save(self.testDataPath, test_data)
                not_continue_count = 0
            else:
                not_continue_count += 1

            print(
                f'{t + 1}\t\t{sum(epoch_loss) / len(epoch_loss):.3f}\t\t{sum(epoch_loss1) / len(epoch_loss1):.3f}\t\t\t{train_srcc:.4f}\t\t{test_srcc:.4f}\t\t{test_plcc:.4f}\t\t{test_krcc:.4f}')

        print(f'Best at epoch {best_epoch}, test SRCC {best_srcc:.4f}, test PLCC {best_plcc:.4f}')
        return best_srcc, best_plcc, best_krcc

    def test(self, data_loader):
        self._net.train(False)
        pscores, tscores = [], []
        batch_size = 128
        test_data = {}

        for X, y, z, path, _ in data_loader:
            X, y, z = X.cuda(), y.cuda(), z.cuda()
            X_sub = self.unfold(X).view(1, X.shape[1], 224, 224, -1)[0].permute(3, 0, 1, 2)
            img = torch.split(X_sub, batch_size, dim=0)
            pred_s = [self._net(i, z)[0].detach().cpu().item() for i in img]
            score = np.mean(pred_s)
            pscores.append(score)
            tscores.extend(y.cpu().tolist())
            test_data[path] = [score, y.cpu().tolist()[0]]

        test_srcc, _ = stats.spearmanr(pscores, tscores)
        test_plcc, _ = stats.pearsonr(pscores, tscores)
        test_krcc, _ = stats.kendalltau(pscores, tscores)
        self._net.train(True)
        return test_srcc, test_plcc, test_data, test_krcc


class flushfile:
    def __init__(self, f):
        self.f = f

    def write(self, x):
        self.f.write(x)
        self.f.flush()


def main():
    parser = argparse.ArgumentParser(description='Test CLIP for IQA tasks.')
    parser.add_argument('--base_lr', type=float, default=1e-4, help='Base learning rate for training.')
    parser.add_argument('--batch_size', type=int, default=64, help='Batch size.')
    parser.add_argument('--epochs', type=int, default=65, help='Epochs for training.')
    parser.add_argument('--weight_decay', type=float, default=5e-4, help='Weight decay.')
    parser.add_argument('--dataset', type=str, default='CSIQ', help='Dataset: CSIQ')
    parser.add_argument('--model', type=str, default='AdmpIQA', help='Model: AdmpIQA')
    parser.add_argument('--n_ctx', type=int, default=8, help='Prompt length')
    parser.add_argument('--n_blocks', type=int, default=3, help='VisualFeatureAdapter: number of ResidualBlock')
    parser.add_argument('--gpuid', type=str, default='0', help='GPU ID')
    parser.add_argument('--percentage', type=float, default=0.8, help='Training portion')
    parser.add_argument('--cfgname', type=str, default='config_8', help='Configuration for prompting learning')
    args = parser.parse_args()
    os.environ['CUDA_VISIBLE_DEVICES'] = args.gpuid

    with open(f'./config/{args.cfgname}.yaml') as f:
        cfg = yaml.load(f, Loader=yaml.FullLoader)

    # Set random seeds for reproducibility
    seed = 10
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

    # Set up logging
    ss = f'{args.model}_{args.dataset}_use_aux_{cfg["TRAINER"]["Ours"]["use_aux"]}_textdepth_{cfg["TRAINER"]["Ours"]["text_depth"]}_visiondepth_{cfg["TRAINER"]["Ours"]["vision_depth"]}_lamda_{cfg["TRAINER"]["Ours"]["lamda"]:.2f}'
    log_path = os.path.join('./log/', 'aux', f'{args.dataset}_{ss}.log')
    with open(log_path, 'w') as f:
        sys.stdout = flushfile(f)
        print(ss)
        print(f"Random Seed: {seed}")

    # Validate input arguments
    if args.base_lr <= 0:
        raise ValueError('--base_lr parameter must be greater than 0.')
    if args.batch_size <= 0:
        raise ValueError('--batch_size parameter must be greater than 0.')
    if args.epochs < 0:
        raise ValueError('--epochs parameter must be non-negative.')
    if args.weight_decay <= 0:
        raise ValueError('--weight_decay parameter must be greater than 0.')

    # Set up save path
    save_path = os.path.join('./save/', args.model)
    os.makedirs(save_path, exist_ok=True)

    # Set up options and paths
    options = {
        'base_lr': args.base_lr,
        'batch_size': args.batch_size,
        'epochs': args.epochs,
        'weight_decay': args.weight_decay,
        'dataset': args.dataset,
        'model': args.model,
        'savePath': save_path,
        'n_ctx': args.n_ctx,
        'cfgname': args.cfgname
    }

    path = {
        'LIVE': 'IQA_Datasets/live/',
        'LIVEC': 'IQA_Datasets/livec/',
        'CSIQ': 'IQA_Datasets/csiq/',
    }

    # Initialize indices for training and testing
    if options['dataset'] == 'CSIQ':
        index = list(range(6))
    elif options['dataset'] == 'Livec':
        index = list(range(6))
    elif options['dataset'] == 'Live':
        index = list(range(5))

    # Initialize metrics
    roudNum = 12
    srcc_all = np.zeros((1, roudNum), dtype=np.float64)
    plcc_all = np.zeros((1, roudNum), dtype=np.float64)
    krcc_all = np.zeros((1, roudNum), dtype=np.float64)

    for i in range(roudNum):
        print(f"====================round {i}=====================")
        random.shuffle(index)
        train_index = index[:int(args.percentage * len(index))]
        test_index = index[int(args.percentage * len(index)):]

        options['train_index'] = train_index
        options['test_index'] = test_index
        options['round'] = i

        manager = IQAManager(options, path, args.percentage, i)
        best_srcc, best_plcc, best_krcc = manager.train()
        srcc_all[0, i] = best_srcc
        plcc_all[0, i] = best_plcc
        krcc_all[0, i] = best_krcc

    # Calculate mean and median metrics
    srcc_mean = np.mean(srcc_all)
    plcc_mean = np.mean(plcc_all)
    krcc_mean = np.mean(krcc_all)
    median_srcc = np.median(srcc_all)
    median_plcc = np.median(plcc_all)

    print('srcc', srcc_all)
    print('plcc', plcc_all)
    print('krcc', krcc_all)

    print(f'average mean srcc: {srcc_mean:.4f}, plcc: {plcc_mean:.4f}, krcc: {krcc_mean:.4f}')
    print(f'average std srcc: {srcc_all.std():.4f}, plcc: {plcc_all.std():.4f}, krcc: {krcc_all.std():.4f}')
    print(f'median srcc: {median_srcc:.4f}, median plcc: {median_plcc:.4f}')


if __name__ == '__main__':
    main()
