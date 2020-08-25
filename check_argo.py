import argoverse
from argoverse.data_loading.argoverse_tracking_loader import ArgoverseTrackingLoader
import pdb
pdb.set_trace()
root_dir = '/ssd_scratch/cvit/raghava.modhugu/argoverse-tracking/'
argoverse_loader = ArgoverseTrackingLoader(root_dir)
camera = 'ring_front_center'
with open('splits/argoverse/val_files.txt') as f:
   for cnt, line in enumerate(f):
       log, index, _ = line.split()
       print(log, index)
       print(argoverse_loader.get(log).get_image(int(index), camera = camera, load=False))
