import os
root = '/ssd_scratch/cvit/raghava.modhugu/argoverse-tracking/'
log_ids = [ i for i in os.listdir(root) if i not in ['test', 'train', 'val']]
num_log_ids = len(log_ids)
with open('splits/argoverse/files.txt', 'w') as f:
     for log in log_ids:
         log_path = os.path.join(root, log, 'ring_front_center')
         for file in range(len(os.listdir(log_path))):
             f.writelines('{} {} {}\n'.format(log, file, 'c'))
