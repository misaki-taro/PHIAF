import argparse
import os
import pickle

######################################
######## INPUT PARAMETERS ############
######################################
parser = argparse.ArgumentParser(description='Test Argparse')
parser.add_argument('--name', default='misaki', help='test name')
parser.add_argument('--your-email', default='1234@qq.com', help='test email')

inputs = parser.parse_args()

# Print configurations
print('######################################################')
print('######################################################')
print('Name:                {0}'.format(inputs.name))
print('Email:               {0}'.format(inputs.your_email))

aa = [1,2,3]

for k, v in inputs.__dict__.items():
    print(k, v)

save_root_dir = '../tune_results/'
sub_dir = 'testarg'
os.makedirs(save_root_dir+sub_dir, exist_ok=True)

file_name = ''
for k, v in inputs.__dict__.items():
    file_name += '{0}-{1}__'.format(k, v)

with open('{0}{1}/{2}.pkl'.format(save_root_dir, sub_dir, file_name), 'wb') as f:
    pickle.dump(aa, f)
    print('Saved!')