# Forward model simulation
# output is 2 .txt files with with dipoles_locations_and_values and sensor_locations_and_values
import os
from copy import deepcopy
import numpy as np
import pandas as pd
#from scipy.spatial.distance import cdist
# import pickle as pkl
#import dill as pkl
import random
#from joblib import Parallel, delayed
# from tqdm.notebook import tqdm
from tqdm import tqdm
from mne.channels.layout import _find_topomap_coords
#import colorednoise as cn
import mne

from .raz_util import *



DEFAULT_SETTINGS = {
	'method': 'Parcellation', #or"standart" hecker
	'number_of_labels': 'all', #15
	'number_of_sources': (1, 25),
	#'extents':  (1, 50),  # in millimeters
	
	'amplitudes': (1e-3, 100),

	#'shapes': 'mixed',

	#'duration_of_trial': 1.0,
	#'sample_frequency': 100,
	'target_snr': (1, 20),

	'source_spread': "mixed",
	'source_number_weighting': True,
	#'source_time_course': "random",
}


class Simulation:
	''' Simulate and hold source and EEG data.
	
	Attributes
	----------
	settings : dict
		The Settings for the simulation. Keys:

		number_of_sources : int/tuple/list
			number of sources. Can be a single number or a list of two numbers specifying a range.
		amplitudes : int/float/tuple/list
			the current of the source in nAm 
		target_snr : float/tuple/list
			The desired average SNR of the simulation(s)

	fwd : mne.Forward
		The mne-python Forward object that contains the forward model
	source_data : mne.sourceEstimate
		A source estimate object from mne-python which contains the source 
		data.
	eeg_data : mne.Epochs
		A mne.Epochs object which contains the EEG data.
	
	Methods
	-------
	simulate : Simulate source and EEG data
	plot : plot a random sample source and EEG

	'''
	def __init__(self, fwd, info, trans, subjects_dir, 
				 save_dir,
				 settings=DEFAULT_SETTINGS, 
				 #n_jobs=-1, parallel=False, verbose=False):
				 ):
		self.settings = settings
		self.save_dir = save_dir
		
		self.source_data = None
		self.eeg_data = None

		self.ch_names = info.ch_names
		self.info = deepcopy(info) #temp variable

		self.trans = deepcopy(trans)


		self.fwd = deepcopy(fwd)
		self.fwd.pick_channels(info['ch_names']) #only channels in raw.info file


		self.subject = self.fwd['src'][0]['subject_his_id']

		self.subjects_dir = subjects_dir
		os.environ['SUBJECTS_DIR']= subjects_dir

		#self.check_info(deepcopy(info))
		self.check_settings()
		#self.prepare_simulation_info()

		#self.n_jobs = n_jobs
		#self.parallel = parallel
		#self.verbose = verbose

	def check_settings(self):
		''' Check if settings are complete and insert missing entries if there are any.
		'''
		if self.settings is None:
			self.settings = DEFAULT_SETTINGS
		self.fwd_fixed, self.leadfield, self.dip_pos, self.vert = unpack_fwd(self.fwd)

		
		self.eeg_pos_trans = get_eeg_pos(self.info, self.trans, self.subject, self.subjects_dir, scale=1000)

		all_dip = np.vstack((self.dip_pos[0], self.dip_pos[1]))
		self.dip_pos_trans = get_dipole_pos(all_dip, self.trans, scale=1000)
		#self.distance_matrix = cdist(all_dip, all_dip)




	# not ready
	def simulate(self, n_samples=5, save=True):
		''' Simulate sources and EEG data'''
		self.n_samples = n_samples

		if self.settings['method'] == "Parcellation" :

			n_labels = self.settings['number_of_labels']
			sim_type = self.settings['method'] + f"_{n_labels}-lbl"

			file_list = []

			for i in range(n_samples):
				source_data = self.simulate_label_source(n_labels=n_labels, return_stc=False)
				eeg_data = self.simulate_eeg(source_data, scale=1, return_evk=False) 
				#print(f"\n eeg_data min={eeg_data.min() }")      
				#print(f"\n eeg_data max={eeg_data.max() }")
				#print(len(np.unique(eeg_data)))
				if save:
					save_to = self.save_dir+"/"+sim_type
					os.makedirs(save_to, exist_ok=True)

					file_name = r"/{}-lbl_{}-{}.npy"

					fname_dip = save_to + file_name.format(n_labels, "dip", i)
					#print(f"\nfname_dip:\n {fname_dip}")
					save_posVal(fname_dip, self.dip_pos_trans, source_data, norm=True)

					fname_eeg = save_to + file_name.format(n_labels,"eeg", i)
					#print(f"\nfname_eeg:\n {fname_eeg}")
					save_posVal(fname_eeg, self.eeg_pos_trans, eeg_data, norm=True)

					file_list.append((fname_eeg, fname_dip))
			return file_list
			#pass

		
	def read_parcellation(self, parcellation='aparc.a2009s', hemi='both'):

		labels = mne.read_labels_from_annot(self.subject, parc=parcellation, hemi=hemi, surf_name='white', 
												  subjects_dir=self.subjects_dir, sort=True,
												  annot_fname=None, regexp=None)
		if parcellation=='aparc.a2009s':
			labels = labels[:-2]  #exclude median wall 
		return labels

	def simulate_eeg(self, source_data, scale=1e-9, return_evk=False):
		eeg_data = np.matmul(self.leadfield, source_data*scale)

		# add eeg_noise block here

		if return_evk:
			evk = mne.EvokedArray(eeg_data, self.info, 
								  tmin=0, nave=1, comment='simulated')
			return evk
		else:
			return eeg_data


	def simulate_label_source(self, n_labels="all", return_stc=False):
		'''pick vertices in a label and randomly set current values'''

		labels = self.read_parcellation()
		N_labels = len(labels)
		label_list=[]

		n_ch, n_dip = self.leadfield.shape
		data = np.zeros((n_dip,1))


		if n_labels=='all':
			# assign Gaussian distribution to all labels
			label_id =  np.arange(0, N_labels)
		elif n_labels>0:
			# pick labels subset
			label_id = np.random.randint(0, N_labels, n_labels)


		for i, lbl_id in enumerate(label_id):
			lbl_cur = labels[lbl_id]
			label_list.append(lbl_cur)

			# get indices to assign data[ind] 
			hemi = lbl_cur.name.split("-")[-1]
			if hemi=='rh':
				n_dipoles_lh = self.vert[0].shape

				v = lbl_cur.get_vertices_used(vertices=self.vert[1])               
				vert = self.vert[1]        
				v_stc_id = np.nonzero(np.in1d(vert, v))[0]
				v_stc_id += n_dipoles_lh        
			else:
				vert = self.vert[0]
				v = lbl_cur.get_vertices_used(vertices=self.vert[0])
				v_stc_id = np.nonzero(np.in1d(vert, v))[0]

			# generate signal as gaussian distribution
			mean = np.random.random_integers(35, 135) #raz_check
			var_scale = np.random.uniform(0.5, 25)	 #raz_check
			current_values = np.random.normal(loc=mean, scale=var_scale, size=(len(v_stc_id),1))
		   
			data[v_stc_id,:] = current_values

		# add source-noise block here

		if return_stc:
			tmin=0; sfreq=200

			all_vert = [self.vert[0], self.vert[1]]
			stc = mne.SourceEstimate(data, all_vert, subject=self.subject,
									 tmin=tmin, tstep=1/sfreq, )
			return stc
		else:
			return data