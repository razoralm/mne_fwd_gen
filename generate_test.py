#import nibabel as nib
import os
import numpy as np
import pyvista as pv
#import pandas as pd
#import sklearn


def pv_viz(eeg_file, dip_file, head_file):

	grid = pv.UnstructuredGrid(head_file)

	eeg = np.load(eeg_file)
	dipole = np.load(dip_file)

	# eeg - red colors
	eeg_data = eeg[:,-1]
	eeg_data = eeg_data[:, np.newaxis]
	z = np.zeros(eeg_data.shape)
	eeg_data = 255*np.concatenate((eeg_data, z, z), axis=1)

	#blue colours
	dipole_data = dipole[:,-1]
	dipole_data = dipole_data[:,np.newaxis]
	z = np.zeros(dipole_data.shape)
	dipole_data = 255*np.concatenate((z, z, dipole_data), axis=1)

	# render
	eeg_mesh = pv.PolyData(eeg[:,:-1])
	eeg_mesh["colors"] = eeg_data.astype(np.uint8)

	dipole_mesh = pv.PolyData(dipole[:,:-1])
	dipole_mesh["colors"] = dipole_data.astype(np.uint8)

	pl = pv.Plotter()
	pl.add_mesh(grid, opacity=0.25, color='w')
	pl.add_mesh(eeg_mesh, scalars="colors", rgb=True)
	pl.add_mesh(dipole_mesh, scalars="colors", rgb=True)
	pl.show()

from gen import raz_simulation

import mne
from mne.viz import plot_alignment, set_3d_view, plot_montage
print(mne.__version__)
print(mne.__path__)
mne.set_log_level('CRITICAL')
#mne.set_log_level('WARNING')

subjects_dir = r'C:/vm_shared/freesurfer/'

eeg_dir = r'C:/vm_shared/EEG_processing/'
trans_dir = r'C:/vm_shared/TRANS_data/'
fwd_path = r'C:/vm_shared/FWD/'

gen_dir =  r'C:/vm_shared/MNE_GEN/'

fwd_template = "/{subj}_{eeg_layout}_{source_spacing}-dip_{bem_spacing}-bem.fif"

head_template = "C:/vm_shared/vtk_mfem/{subj}_head_mfem.vtk"

subj_dict = {"AR": {"raw": r'chess/AR_chess_30Hz_eeg.fif',
			   "trans": 'Razorenova-eeg1005-trans.fif',
			   "subj":  'Razorenova',
			   "mfem":  'AR_head_mfem.vtk'              
			   },
		 
		 "AB": {"raw": r'chess/AB_chess_30Hz_eeg.fif',
			   "trans": 'Butorina-eeg1005-trans.fif',
			   "subj":  'Butorina',
			   "mfem":   'AB_head_mfem.vtk'
			   },
				  
		 "ES": {"raw":  r'chess/ES_chess_30Hz_eeg.fif',
			   "trans": 'SKidchenko-eeg129-trans.fif',
			   "subj":  'SKidchenko',
			   "mfem":   'ES_head_mfem.vtk'
			   },
		 
		 "GK": {"raw": r'chess/GK_chess_30Hz_eeg.fif',
			   "trans": 'newGK_audiovis-trans.fif',
			   "subj":  'Kormakov',
			   "mfem":  'GK_head_mfem.vtk'
			   },
		 
		 "NK": {"raw": r'tone1000Hz/NK_tone1000Hz_30Hz_eeg.fif',
			   "trans": 'Koshev-eeg1005-trans.fif',
			   "subj" : 'NKoshev',
			   "mfem":  'NK_head_mfem.vtk'
			   },
		}


#debug
#subj_dict = {"AR": {"raw": r'chess/AR_chess_30Hz_eeg.fif',
#			   "trans": 'Razorenova-eeg1005-trans.fif',
#			   "subj":  'Razorenova'                
#			   },}


n_samples = 2

SIM_SETTINGS = {
	'method': 'Parcellation', #or"standart" hecker
	'number_of_labels': 2 ,   #'all', #15
}



####
for subject, sub_dict in subj_dict.items():

	layout = "128-eeg1005"
	spacing = "ico4"

	# get info
	eeg_path = eeg_dir + sub_dict["raw"]
	eeg_raw = mne.io.read_raw_fif(eeg_path, preload=True).pick_types(eeg=True)
	eeg_raw.set_eeg_reference(projection=True)
	eeg_raw=eeg_raw.interpolate_bads(reset_bads=True, mode='accurate', origin='auto')
	print(f"eeg_path: {eeg_path}")

	#read trans
	trans_path = trans_dir + sub_dict["trans"]
	trans = mne.read_trans(trans_path)
	print(f"trans_path: {trans_path}")

	# get fwd
	fwd_fname = fwd_template.format(subj = sub_dict["subj"],
									eeg_layout = layout,
									source_spacing = spacing,
									bem_spacing = "ico4")

	fwd_file = fwd_path + sub_dict["subj"] + fwd_fname
	fwd=mne.read_forward_solution(fwd_file)
	print(f"fwd_path: {fwd_fname}")
	
	print(f"\n {subject} files_read")


	# simulate data
	save_dir = gen_dir + fwd_fname.replace(".fif","/")
	 
	sim = raz_simulation.Simulation(fwd, eeg_raw.info, trans, subjects_dir, save_dir=save_dir,
									settings=SIM_SETTINGS)

	file_pairs = sim.simulate(n_samples=n_samples)
	
	pv_check = True
	if pv_check:
		head_file = head_template.format(subj=subject)
		for eeg_file, dip_file in file_pairs:
			pv_viz(eeg_file, dip_file, head_file)

	print(f"\n {n_samples} eeg-dip generated to {save_dir}")
	if SIM_SETTINGS['method']=='Parcellation':
		print(f"  method: {SIM_SETTINGS['method']}\tfilled_labels: {SIM_SETTINGS['number_of_labels']}")