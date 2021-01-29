# Reorders Gadget snapshots into sorted HDF5 file

import numpy as np
import h5py
import math
import time,sys,os
import datetime
import readsnap
from mpi4py import MPI
import resource

verbose = False
DEBUG = 2
GADGET = False

ptype = 1 #dark matter
# Set folder for sorted snapshots
#output_path = '/cosma7/data/dp004/dc-boot5/Ordered_Snapshots/Npart_512_Box_750-Fiducial/'
output_path = '/cosma7/data/dp004/dc-boot5/Ordered_Snapshots/Npart_2048_Box_3000-Fiducial/'

#define particle datatype
vect = np.dtype([('x', np.float32),('y', np.float32),('z', np.float32)])
part = np.dtype([('pos', vect),('vel', vect),('ID', np.ulonglong)])

#####################################################################################
# This function reads a sub-snapshot file in GADGET binary format and reorders by ID
#####################################################################################
def read_subsnapshot(fnum, block):
	# find the name of the sub-snapshot
	snapshot = snapshot_fname + '.{0:d}'.format(fnum)

	# find the local particles in the sub-snapshot
	head  = readsnap.snapshot_header(snapshot)
	npart = head.npart

	if verbose:  print ('Sub-snapshot {0:d}, DM particles = {1:d} \n'.format(fnum, npart[ptype]))
	#if (DEBUG>1 and fnum%10 == 0):  print ('Task: {3:d}, Sub-snapshot {0:d}, DM particles = {1:d}, time = {2:%H:%M:%S}'.format(fnum, npart[ptype], datetime.datetime.now(), rank))

	# read particle IDs
	ids = readsnap.read_block(snapshot, "ID  ", ptype)

	# read positions in Mpc/h
	pos = readsnap.read_block(snapshot, "POS ", ptype)

	# read velocities in km/s 
	vel = readsnap.read_block(snapshot, "VEL ", ptype)

	# Assign particle parameters to particle array in ID order
	f = tuple([(ids > block * Nparts) & (ids <= (block+1) * Nparts)])
		
	i = ids[f] - (block * Nparts)
	ordered_parts['ID'][i] = ids[f]
	ordered_parts['pos']['x'][i] = pos[::,0][f]
	ordered_parts['pos']['y'][i] = pos[::,1][f]
	ordered_parts['pos']['z'][i] = pos[::,2][f]
	ordered_parts['vel']['x'][i] = vel[::,0][f]
	ordered_parts['vel']['y'][i] = vel[::,1][f]
	ordered_parts['vel']['z'][i] = vel[::,2][f]
	
#####################################################################################
# Start of program
#####################################################################################

comm = MPI.COMM_WORLD
rank = comm.Get_rank()
size = comm.Get_size()

start_snap = 60
nsnaps = 1
snap = 60

#input file 
snapshot_fname = '/cosma6/data/dp004/dc-smit4/Daemmerung/Planck2013-Npart_2048_Box_3000-Fiducial/run1/snapdir_{0:03d}/Planck2013-L3000-N2048-Fiducial_{0:03d}'.format(snap)	
	
# read snapshot header
head = readsnap.snapshot_header(snapshot_fname + '.0')
filenum = head.filenum
Nparts = head.nall_hi[1] * 2**32 + head.nall[1]

Nparts //=2

if (rank == 0) | (rank == 28): 
	print('Number of particles per block =', Nparts)
	nbytes = (Nparts + 1) * part.itemsize 
else: 
	nbytes = 0

win = MPI.Win.Allocate_shared(nbytes, part.itemsize , comm=comm) 

# create a numpy array whose data points to the shared mem
if (rank<28):
	mb = 0
	block = 0
else:
	mb = 28
	block = 1
	
buf, itemsize = win.Shared_query(mb) 

assert itemsize == part.itemsize 
ordered_parts = np.ndarray(buffer=buf, dtype=part, shape=(Nparts+1,)) 

# read in each sub-snapshot file
chunk = size//2  # this is number of cores participating in ths file read operation
brank = rank - mb
batch = filenum//chunk
rem = filenum%chunk
start = brank * batch
if (brank == chunk-1):
	end = start + batch + rem
else:
	end = start + batch 
	
for i in range(start,end):  		
	if (DEBUG>1) & ((i==start) | (i==end-1)):  print ('Task: {0:d}, Sub-snapshot {1:d}, time = {2:%H:%M:%S}'.format(rank, i, datetime.datetime.now()))
	read_subsnapshot(i, block)

# wait for all tasks to complete
comm.Barrier()
if (DEBUG>0 and rank==0):  print ('Completed processing {0:d} snapshots, time = {1:%H:%M:%S}'.format(nsnaps, datetime.datetime.now()))

# determine memory segment to be written to file by this task
segsize = Nparts//chunk
rem = Nparts%chunk
offset = segsize * brank
if (brank==chunk-1):
	last = offset + segsize + rem
else:
	last = offset + segsize
	
res = resource.getrusage(resource.RUSAGE_SELF)
#print('Task: {0:d}, max rss = {1:0d}, shared mem size = {2:0d}'.format(rank, res.ru_maxrss,res.ru_ixrss))

if (rank>=0):
	ordered_snap = output_path + 'snapdir_{0:02d}/ordered_snapshot.snap_{0:03d}.{1:02d}.hdf5'.format(snap, rank)
	with h5py.File(ordered_snap,'w') as fo:
	
		fo.create_dataset('Parts',data = ordered_parts[offset:last])

		# Set header attributes
		h = fo.create_dataset('/Header', dtype = 'f')
		h.attrs.create('Time', head.time)
		h.attrs.create('Redshift', head.redshift)
		h.attrs.create('BoxSize', head.boxsize)
		h.attrs.create('NumFilesPerSnapshot', 1)
		h.attrs.create('Omega0', head.omega_m)
		h.attrs.create('OmegaLambda', head.omega_l)
		h.attrs.create('HubbleParam', head.hubble)
		h.attrs.create('MassTable', head.massarr)
		h.attrs.create('NumPart_ThisFile', head.npart)
		h.attrs.create('NumPart_Total', head.nall + head.nall_hi * 2**32)
		h.attrs.create('Flag_Cooling', head.cooling)

	if (DEBUG>0):  print ('Task: {0:0d} - ordered snapshots written to file, offset = {1:0d}, last = {2:0d}, time = {3:%H:%M:%S}'.format(rank, offset, last, datetime.datetime.now()))
		