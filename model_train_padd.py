import os
import matplotlib.pyplot as plt
import configparser
from WavLoc import WavLoc
# import time

def main():
	# basic variables
	train_set_dir_base = '../Data/v1/train/record'
	valid_set_dir_base = '../Data/v1/valid/record/'

	model_dir = 'models/padd'
	rooms = ['Anechoic','A','B','C','D']
	for tar_room in rooms[1:]:
		model_dir_room = os.path.join(model_dir,'mct_not_{}'.format(tar_room))
		if not os.path.exists(model_dir_room):
			os.makedirs(model_dir_room)

		# filter out tar_room from rooms
		mct_room_filter = lambda room : room != tar_room

		config = configparser.ConfigParser()
		config['model']={'model_dir':model_dir_room,
		                'frame_len':320,
						'overlap_len':160,
		                'filter_len':320,
						'is_use_gtf':False,
						'is_padd':True,
		                'azi_num':37}
		config['train']={'batch_size':128,
		                 'max_epoch':50,
						 'is_print_log':True,
		                 'train_set_dir':';'.join([os.path.join(train_set_dir_base,room)
													 for room in filter(mct_room_filter,rooms)]) ,
		                 'valid_set_dir':';'.join([os.path.join(valid_set_dir_base,room)
													 for room in filter(mct_room_filter,rooms)])}

		config_fpath = os.path.join(model_dir_room,'mct_not_{}.cfg'.format(tar_room))
		with open(config_fpath,'w') as config_file:
		    if config_file is None:
		        raise Exception('fail to create file')
		    config.write(config_file)

		model = WavLoc(config_fpath,gpu_index=0)
		model.train_model()


if __name__ == '__main__':
    main()
