import pyvista as pv
import numpy as np
import mne

#hacker-raz
def unpack_fwd(fwd): 
	""" Helper function that extract the most important data structures from the 
	mne.Forward object

	Parameters
	----------
	fwd : mne.Forward
		The forward model object

	Return
	------
	fwd_fixed : mne.Forward
		Forward model for fixed dipole orientations
	leadfield : numpy.ndarray
		The leadfield (gain matrix)
	pos : numpy.ndarray
		The positions of dipoles in the source model
	tris : numpy.ndarray
		The triangles that describe the source mmodel
	"""
	fwd_fixed = fwd
	if not fwd['surf_ori']:
		print("Forward model does not contain fixed orientations - expect unexpected behavior!")
		fwd_fixed = mne.convert_forward_solution(fwd, surf_ori=True, force_fixed=True, 
													use_cps=True, verbose=0)
 
	leadfield = fwd_fixed['sol']['data']
	print("Leadfield size : %d sensors x %d dipoles" % leadfield.shape)

	source_model = fwd_fixed['src']

	#dipole info
	n_dipoles_lh = int(source_model[0]['nuse'])
	n_dipoles_rh = int(source_model[1]['nuse'])
	number_of_dipoles = n_dipoles_lh+n_dipoles_rh

	#vert indices
	vertices_hemi = [source_model[0]['vertno'], source_model[1]['vertno']]

	#vert coords
	vert_pos_hemi = [source_model[0]['rr'][source_model[0]['vertno']],
		source_model[1]['rr'][source_model[1]['vertno']]]

	vert_pos = np.vstack((vert_pos_hemi[0], vert_pos_hemi[1]))
	#print(len(vert_pos)==number_of_dipoles)


	return fwd_fixed, leadfield, vert_pos_hemi, vertices_hemi#tris#


def get_eeg_pos(info, trans, subject, subjects_dir,
				surf=r"/bem/outer_skin.surf", scale=1): 
	'''apply transform  head-->MRI-coordinates to eeg-sensors locations'''

	surf_path = subjects_dir+subject+surf
	#print(f"\t{surf_path}")
	surf = mne.surface.read_surface(surf_path, return_dict=True)[-1]

	## ideal electrodes position (head axes), [m]
	eeg_loc = np.array([info['chs'][k]['loc'][:3] for k in range(info['nchan'])])
	#print(f"orig {eeg_loc[0]}")
	## apply trans
	eeg_loc_trans = mne.transforms.apply_trans(trans, eeg_loc)
	#print(f"orig_trans {eeg_loc_trans[0]}")
	# project on surface [mm]
	eeg_loc_trans, eeg_nn_trans = mne.surface._project_onto_surface(eeg_loc_trans*scale, surf, 
														  project_rrs=True, return_nn=True)[2:4]
	#print(f"out_trans {eeg_loc_trans[0]}")
	return eeg_loc_trans


def get_dipole_pos(dip_pos, trans, scale=1): 
	'''apply transform  head-->MRI-coordinates to dipoles'''
	dip_pos_trans = mne.transforms.apply_trans(trans, dip_pos)
	return dip_pos_trans*scale


def norm_data(data):
	#print(f"\n amin = {np.amin(data)} \n amax = {np.amax(data)}")
	norm_data = (data-np.amin(data))/(np.amax(data)-np.amin(data))
	return norm_data


def save_posVal(fname, pos, val, norm=True):
	#print("pos",pos.shape, pos[0])
	#print("val", val.shape, val[0])

	#val = np.squeeze(val)
	if norm:
		data = norm_data(val) 
	np_data = np.hstack((pos, data))

	#print("save to:", fname)
	np.save(fname, np_data)



