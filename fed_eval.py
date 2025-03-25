import os
import sys
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
os.environ["KMP_DUPLICATE_LIB_OK"] = 'True'
sys.path.append(os.path.dirname(os.path.realpath(__file__)))
sys.path.append(os.path.dirname(os.path.dirname(os.path.realpath(__file__))))
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.realpath(__file__)))))
import argparse
import pandas as pd
import numpy as np
import torch
from models import deeplabv3
from tqdm import tqdm
from fedgraph.dataset import get_dataloader
from torch.nn.functional import one_hot
import torchmetrics
from models.unet2d import UNet2D
from models.unet34 import Model
from models.unetFedirm import UNet
from utils.metirc import MultDice, com_hd, ASSD
from utils.utils import write_csv
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# 联邦学习中的验证

def get_args(known=False):
    parser = argparse.ArgumentParser(description='PyTorch Implementation')
    parser.add_argument('--seed', type=int, default=2001, metavar='S', help='random seed (default: 1)')
    parser.add_argument('--client', type=int, default=0, metavar='S', help='random seed (default: 1)')
    # parser.add_argument('--project', type=str, default='/media/pan/Lab-pdl/pfssl-exp/runs/UCMT', help='project path for saving results')
    parser.add_argument('--project', type=str, default='./runs/UCMT')
    parser.add_argument('--backbone', type=str, default='UNet', choices=['DeepLabv3p', 'UNet'], help='segmentation backbone')
    parser.add_argument('--data_path1', type=str, default='G:\\new-server-data\\UCMT\dataset\polyp-all', help='path to the data')
    parser.add_argument('--data_path2', type=str, default='./dataset/liver2_all_data',
                        help='path to the data')
    parser.add_argument('--is_cutmix', type=bool, default=False, help='cut mix')
    parser.add_argument('--labeled_percentage', type=float, default=0.3, help='the percentage of labeled data')
    parser.add_argument('--image_size', type=int, default=128, help='the size of images for training and testing')
    parser.add_argument('--batch_size', type=int, default=1, help='number of inputs per batch')
    parser.add_argument('--num_workers', type=int, default=1, help='number of workers to use for dataloader')
    parser.add_argument('--in_channels', type=int, default=3, help='input channels')
    parser.add_argument('--data_parties', type=int, default=5, help='data_parties nums')
    parser.add_argument('--num_classes', type=int, default=2, help='number of target categories')
    parser.add_argument('--model_weights', type=str, default='best.pth', help='model weights')
    parser.add_argument('--ssl', type=str, default='sup', choices=['ours', 'bcp', 'mt', 'sup'], help='ssl function')
    parser.add_argument('--dataset_type', type=str, default='json', choices=['json', 'random'], help='read dataset type')
    args = parser.parse_known_args()[0] if known else parser.parse_args()
    return args

def load_model(model_weights, in_channels, num_classes, backbone):
    # model = deeplabv3.__dict__[backbone](in_channels=in_channels, out_channels=num_classes).to(device)
    # model = UNet2D(3, 2).to(device)
    model = Model(3, 2).to(device)
    # model = UNet(3,2).to(device)
    # print('#parameters:', sum(param.numel() for param in model.parameters()))
    # print(model_weights["state_dict"])
    model.load_state_dict(torch.load(model_weights))
    return model

def fed_eval():
    args = get_args()
    # Project Saving Path
    project_path = args.project + '_{}_{}_{}/'.format(args.backbone, args.ssl, str(args.seed))  # 新版本
    # project_path = args.project + '_{}_{}/_{}_{}'.format(args.backbone, str(args.seed), args.backbone, str(args.seed))
    # project_path = args.project
    test_csv_path = os.path.join(project_path, 'eval', 'eval.csv')
    write_csv(test_csv_path, "net_id", ["dice", "miou", "acc", "hd95"])

    # get data
    data_parties = args.data_parties
    local_dls, test_dl = get_dataloader("json", args.data_path1, args.data_path2, args.image_size,
                                        args.batch_size, data_parties, args.labeled_percentage)

    test_miou_avg_list = []
    test_dice_avg_list = []
    test_hd95_avg_list = []
    test_auroc_avg_list = []
    test_assd_avg_list = []
    test_std_avg_list = {'miou':[], 'dice':[], 'auc':[], 'hd95':[], 'assd':[]}
    print('start evaluation')
    for net_id in range(0, data_parties):
        # if net_id == 1:
        #     continue
        # Load Data
        _, _, test_local_dl = local_dls[net_id]
        # metric
        # test_metric_iou = torchmetrics.JaccardIndex(task="multiclass", num_classes=2).to(device)
        test_metric_iou = torchmetrics.JaccardIndex(task="multilabel", num_labels=2).to(device)  # iou
        # test_metric_acc = torchmetrics.classification.MultilabelAccuracy(num_labels=2).to(device) # todo 废弃
        test_metric_dice = MultDice().to(device)  # dice
        test_metric_auroc = torchmetrics.classification.MultilabelAUROC(num_labels=2, average="macro", thresholds=None).to(device)  # auc/roc
        # test_metric_iou2 = MeanIoU(num_classes=2, per_class=False, include_background=True).to(device)  # todo 废弃
        # test_metric_gds = GeneralizedDiceScore(num_classes=2, per_class=False, include_background=True).to(device)  # todo 废弃d
        test_metric_assd = ASSD().to(device)
        test_hd95_list = []
        test_miou_item_list = []
        test_dice_item_list = []
        test_assd_item_list = []
        test_hd95_item_list = []
        test_auroc_item_list = []
        # Load model    best_test_dice   best_test_iou
        project_path = "./runs/UCMT_UNet_ours_1001"
        weights_path = os.path.join(project_path, str(net_id), 'weights/last.pth')
        # weights_path = os.path.join(project_path, 'best.pth')
        # weights_path = "/home/pan/PDL/UCMT/runs_ab/UCMT_UNet_401/weights/best.pth"
        # weights_path = "/media/pan/Lab-pdl/new-server-data/UCMT/runs_polyp/UCMT_UNet_mt_1003/best.pth"
        # weights_path = "/home/pan/PDL/UCMT/runs_polyp/UCMT_UNet_mt_1003/best.pth"
        # weights_path = "./runs/UCMT_UNet_ours_1001"
        # weights_path = "./runs/best.pth"
        print(weights_path)
        model = load_model(model_weights=weights_path, in_channels=args.in_channels, num_classes=args.num_classes, backbone=args.backbone)
        model.eval()
        ############################
        # Evaluation
        ############################
        with torch.no_grad():
            test_bar = tqdm(test_local_dl)
            for idx, (image, label, img_name) in enumerate(test_bar):
                # image, label, img_name = next(iter_test_dataloader)
                image, label = image.to(device), label.to(device)
                pred_test_data = model(image)  # [1,2,128,128]

                label = label.squeeze(dim=1).long()
                label_hot = one_hot(label, num_classes=2).permute(0, 3, 1, 2).contiguous()
                pred_feature = torch.softmax(pred_test_data, dim=1)  # [1,2,128,128]
                pred_max = torch.argmax(pred_feature, dim=1)  # [1,128,128]
                pred_test = one_hot(pred_max, num_classes=2).permute(0, 3, 1, 2).contiguous()

                # print(pred_test_data.shape, pred_feature.shape, pred_max.shape)

                pred_test_data_np = np.array(image.data.cpu())
                df = pd.DataFrame(pred_test_data_np.reshape(-1, pred_test_data_np.shape[-1]))  # 扁平化数组以便保存为CSV
                # 保存为 CSV 文件
                df.to_csv("image.csv", index=False, header=False)


                # save
                # pred_max_class = torch.argmax(image, dim=1)  # [batch_size, height, width]
                # # 选择第一个样本
                # image_to_save = pred_max_class[0].cpu().numpy()
                # # 绘制并保存图像
                # plt.imshow(image_to_save, cmap='gray')  # 使用灰度颜色映射
                # plt.axis('off')  # 不显示坐标轴
                # plt.savefig('prediction_image444.png', bbox_inches='tight', pad_inches=0)
                # break

                # cal metric
                test_miou = test_metric_iou(pred_test, label_hot)
                test_dice = test_metric_dice(pred_test, label_hot)
                test_auroc = test_metric_auroc(pred_feature, label_hot)
                test_hd95 = com_hd(label, pred_max)
                test_assd = test_metric_assd(pred_test, label_hot)
                test_hd95_list.append(test_hd95)
                # test_hd95 = metrics.compute_hausdorff_distance(pred_test, label_hot, percentile=95)
                # test_hd95_list.append(torch.mean(test_hd95).item())
                # test_asd = com_assd(pred_feature, label_hot)
                # print(test_assd)
                write_csv("./val.csv", 0, [img_name, test_miou, test_dice])

                test_miou_item_list.append(test_miou.item())
                test_dice_item_list.append(test_dice.item())
                test_hd95_item_list.append(test_hd95)
                test_auroc_item_list.append(test_auroc.item())
                test_assd_item_list.append(test_assd.item())

            # save result
            test_miou_avg = test_metric_iou.compute()
            test_dice_avg = test_metric_dice.compute()
            test_auroc_avg = test_metric_auroc.compute()
            test_assd_avg = test_metric_assd.compute()
            test_hd95_avg = sum(test_hd95_list) / len(test_hd95_list)
            test_miou_avg_list.append(test_miou_avg.item())
            test_dice_avg_list.append(test_dice_avg.item())
            test_hd95_avg_list.append(test_hd95_avg)
            test_auroc_avg_list.append(test_auroc_avg.item())
            test_assd_avg_list.append(test_assd_avg.item())

            test_miou_std = np.std(test_miou_item_list)
            test_dice_std = np.std(test_dice_item_list)
            test_auc_std = np.std(test_auroc_item_list)
            test_hd95_std = np.std(test_hd95_item_list)
            test_assd_std = np.std(test_assd_item_list)
            test_std_avg_list['miou'].append(test_miou_std)
            test_std_avg_list['dice'].append(test_dice_std)
            test_std_avg_list['auc'].append(test_auc_std)
            test_std_avg_list['hd95'].append(test_hd95_std)
            test_std_avg_list['assd'].append(test_assd_std)
            sd_list = [test_miou_std, test_dice_std, test_auc_std, test_hd95_std, test_assd_std]
            print(f"sd_list: {sd_list}")

            test_metrics_list = [test_dice_avg.item(), test_miou_avg.item(),
                                 test_hd95_avg, test_auroc_avg.item(), test_assd_avg.item()]
            print(f"client{net_id} , test_metric:{test_metrics_list}")
            write_csv(test_csv_path, net_id, test_metrics_list)

    # save results
    test_sd_list = [np.mean(test_std_avg_list['dice']), np.mean(test_std_avg_list['miou']),
                    np.mean(test_std_avg_list['auc']), np.mean(test_std_avg_list['hd95']),
                    np.mean(test_std_avg_list['assd'])]
    print(f"test_sd_list:{test_sd_list}")
    test_metrics_avg_list = [np.mean(test_dice_avg_list), np.mean(test_miou_avg_list),
                             np.mean(test_hd95_avg_list), np.mean(test_auroc_avg_list),
                             np.mean(test_assd_avg_list)]
    print(f"avg_test_metric:{test_metrics_avg_list}")
    write_csv(test_csv_path, 88, test_metrics_avg_list)
    print('EVAL FINISHED!')

def global_eval():
    pass

if __name__ == '__main__':
    fed_eval()

