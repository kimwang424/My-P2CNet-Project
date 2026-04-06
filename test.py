import os
import argparse
import numpy as np
import torch
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
from models.P2CNet import P2CNet
from datasets.dataloader import TestDataset
from torch.utils.data import DataLoader
from PIL import Image


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--ckpt_path', default='./ckpt/P2C/P2CNet.pth', help="path to the saved checkpoint of model")
    parser.add_argument('--test_path', default='./demo', type=str, help='path to the test set')
    parser.add_argument('--bs_test', default=1, type=int, help='[test] batch size (default: 1)')
    parser.add_argument('--out_path', default='./results', type=str, help='path to the result')
    args = parser.parse_args()
    print(args)

    os.environ["CUDA_VISIBLE_DEVICES"] = "0"

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = P2CNet().to(device)
    model.load_state_dict(torch.load(args.ckpt_path, map_location=torch.device('cpu')))
    model.eval()

    with torch.no_grad():
        if not os.path.exists(args.out_path):
            os.makedirs(args.out_path)
        test_set = TestDataset(args.test_path)
        test_loader = DataLoader(test_set, batch_size=args.bs_test, shuffle=False, num_workers=0, pin_memory=True)
        for i, (raw_img, name) in enumerate(test_loader):
            raw_img = raw_img.to(device)
            out = model(raw_img)['lab_rgb']
            out = out.to(device="cpu").numpy().squeeze()
            out = np.clip(out * 255.0, 0, 255)
            save_img = Image.fromarray(np.uint8(out).transpose(1, 2, 0))

            # --- 专属补丁：强行恢复原图大小 ---
            # 1. 顺藤摸瓜找到您的原图，看一眼它本来到底有多宽(W)、多高(H)
            orig_img_path = os.path.join(args.test_path, str(name[0]))
            orig_w, orig_h = Image.open(orig_img_path).size

            # 2. 把洗好的小图，用最高清的算法(LANCZOS)无损拉伸回原来的长宽！
            save_img = save_img.resize((orig_w, orig_h), Image.Resampling.LANCZOS)
            # --------------------------------------

            save_img.save(os.path.join(args.out_path, str(name[0])))
            print('%d|%d' % (i + 1, len(test_set)))
