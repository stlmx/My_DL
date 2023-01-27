import torch
import torch.nn as nn

class BatchMixup(object):
    def __init__(self, ratio) -> None:
        super().__init__()
        self.ratio = ratio
        
    def __call__(self, img1, img2, label1, label2):
        img = self.ratio * img1 + (1 - self.ratio) * img2
        label = self.ratio * label1 + (1 - self.ratio) * label2
        return img, label
    
class cutmix(nn.Module):
    """
    注意label是否为one-hot的形式
    这里需要注意lam是留下来的部分的面积占比, 引入的部分的占比就是1-lam
    """
    def __init__(self, lam) -> None:
        super().__init__()
        self.lam = lam
        self.ratio = torch.sqrt(1 - lam)
    
    def __call__(self, img1, img2, label1, label2):
        img_w, img_h = img1[-2:]
        # 首先是mask的那个方框的中心点rx, ry和长宽w, h
        rx, ry = int(torch.rand(img_w)), int(torch.rand(img_h))
        rw, rh = self.ratio * img_w, self.ratio * img_h
        # 返回mask框的左上角和右下角的坐标，注意用torch.clip限制范围，不要超出原图的大小
        x1 = int(torch.clip(rx - rw // 2, 0, img_w))
        x2 = int(torch.clip(rx + rw // 2, 0, img_w))
        y1 = int(torch.clip(ry - rh // 2, 0, img_h))
        y2 = int(torch.clip(ry + rh // 2, 0, img_h))
        
        img1[:, :, x1:x2:, y1:y2] = img2[:, :, x1:x2, y1:y2]
        # 更新lam，防止clip带来的误差
        self.lam = 1 - rw * rh / img_w * img_h
        
        label = self.lam * label1 + (1 - self.lam) * label2
        return img1, label
        
        
        