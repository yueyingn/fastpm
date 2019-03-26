-- parameter file
------ Size of the simulation -------- 
-- ncdm.lua n_shell n_side n_p
-- For Testing
nc = 128
boxsize = 1000

-------- Time Sequence ----
-- linspace: Uniform time steps in a
-- logspace: Uniform time steps in loga
time_step = linspace(0.01,0.02, 21)

output_redshifts= {99.0, 49.}  -- redshifts of output

-- Cosmology --
omega_m = 0.307494
omega_ncdm = 0.001404
h       = 0.6774

--ncdm split
m_ncdm = {0.05,0.05,0.05}
n_shell = 10
n_side = 1
lvk = true
every_ncdm = 1

-- Start with a linear density field
-- Power spectrum of the linear density field: k P(k) in Mpc/h units
-- Must be compatible with the Cosmology parameter
read_powerspectrum= "Pcb.txt"
linear_density_redshift = 99.0 -- the redshift of the linear density field.
random_seed= 42
particle_fraction = 1.0

--sort_snapshot = false
--
-------- Approximation Method ---------------
force_mode = "fastpm"

pm_nc_factor = 2

np_alloc_factor= 4.0      -- Amount of memory allocated for particle

remove_cosmic_variance = true

dealiasing_type = "aggressive"

-------- Output ---------------

filename = string.format("srunE_SH%d_NS%d_every%d_proc%d_size%d_lvk%s_rcv%s_dt%s_B0.1", n_shell, n_side, every_ncdm, os.get_nprocs(), nc, lvk, remove_cosmic_variance, dealiasing_type)
--loc = "/global/cscratch1/sd/abayer/fastpm/ncdm/Pncdm_init_test/"
loc = "/global/cscratch1/sd/abayer/trash/"--fastpm/ncdm/Pncdm_init_test/"

-- Dark matter particle outputs (all particles)
write_snapshot = loc .. filename .. "/fastpm" 
-- 1d power spectrum (raw), without shotnoise correction
write_powerspectrum = loc .. filename .. "/powerspec"
--write_fof = loc .. filename .. "/fastpm"
