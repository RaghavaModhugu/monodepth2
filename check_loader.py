from Argoverse_loader import *
from utils import *
from torch.utils.data import DataLoader
from tqdm import tqdm

data_path = '/ssd_scratch/cvit/raghava.modhugu/argoverse-tracking/'
height =  32
width = 32
frame_ids = [0, -1, 1]
batch_size = 12 
num_workers = 4

fpath = os.path.join(os.path.dirname('./'), "splits", 'argoverse', "{}_files.txt")

train_filenames = readlines(fpath.format("train"))
val_filenames = readlines(fpath.format("val"))

train_dataset = ArgoverseDataset(data_path, train_filenames, height,width,
            frame_ids, 4, is_train=True, img_ext='.jpg')

train_loader = DataLoader(train_dataset, batch_size, False,
            num_workers=num_workers, pin_memory=True, drop_last=True)

val_dataset = ArgoverseDataset(data_path, val_filenames, height,width,
            frame_ids, 4, is_train=True, img_ext='.jpg')

val_loader = DataLoader(val_dataset, batch_size, False,
            num_workers=num_workers, pin_memory=True, drop_last=True)
print('Training\n')
for batch_idx, inputs in tqdm(enumerate(train_loader)):
    print('train', batch_idx)

for batch_idx, inputs in tqdm(enumerate(val_loader)):
    print('val', batch_idx)
